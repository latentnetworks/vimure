"""Inference model"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import sktensor as skt
import scipy.special as sp
from scipy.stats import poisson

from .log import setup_logging
from .utils import preprocess, get_item_array_from_subs

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import _deprecate_positional_args


INF = 1e10
DEFAULT_EPS = 1e-12
DEFAULT_BIAS0 = 0.0
DEFAULT_MAX_ITER = 20
DEFAULT_NUM_REALISATIONS = 20


class VimureModel(TransformerMixin, BaseEstimator):
    """
    **ViMuRe**

    Fit a probabilistic generative model to double sampled networks. It returns reliability parameters for the
    reporters (theta), average interactions for the links (lambda) and the estimate of the true and unknown
    network (rho). The inference is performed with a Variational Inference approach.
    
    .. note::This closely follows the scikit-learn structure of classes:
        https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py
    """
    @_deprecate_positional_args
    def __init__(
        self,
        undirected: bool = False,
        mutuality: bool = True,
        convergence_tol: float = 0.1,
        decision: int = 1,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        undirected : boolean
            Whether the network is undirected.
        mutuality : boolean
            Whether to use the mutuality parameter.
        convergence_tol : float
            Controls when to stop the optimisation algorithm (CAVI)
        """

        self.undirected = undirected

        if undirected:
            msg = "Overriding mutuality to False since the network is undirected"
            warnings.warn(msg)
            self.mutuality = False
        else:
            self.mutuality = mutuality

        self.convergence_tol = convergence_tol  # tolerance parameter for convergence
        self.decision = decision  # convergence parameter

        self.verbose = verbose
        self.logger = setup_logging("vm.model.VimureModel", verbose)

    # TODO Minor refactoring: make this function easier to read (High cyclomatic complexity)
    #      https://betterembsw.blogspot.com/2014/06/avoid-high-cyclomatic-complexity.html

    def __str__(self) -> str:
        return super().__str__()
    
    def __repr__(self) -> str:
        return super().__repr__()

    def __check_fit_params(
        self,
        X: np.ndarray,
        lambda_prior=(10.0, 10.0),
        theta_prior=(0.1, 0.1),
        eta_prior=(0.5, 1.0),
        rho_prior=None,
        seed: int = None,
        **extra_params,
    ):

        available_extra_params = [
            "R",
            "EPS",
            "K",
            "bias0",
            "max_iter",
            "alpha_lambda",
            "beta_lambda",
            "alpha_teta",
            "beta_teta",
            "num_realisations",
        ]
        for extra_param in extra_params:
            if extra_param not in available_extra_params:
                msg = "Ignoring unrecognised parameter %s." % extra_param
                self.logger.warn(msg)

        # If the network is undirected, then do not estimate the mutuality
        if self.undirected and not np.array_equal(X, np.transpose(X, axes=(0, 2, 1, 3))):
            msg = "If undirected is True, the given network has to be symmetric wrt l and m!"
            self.logger.error(msg)
            raise ValueError(msg)

        # TODO: Instead of having these variables (data_T and data_T_vals),
        #       why not just use X with the appropriate (l,j,i,m) mapping?
        if isinstance(X, np.ndarray) or isinstance(X, skt.dtensor):  # if data is dense array
            if self.mutuality:
                # to use mutuality
                self.data_T = np.einsum("aijm->ajim", X)
                # to calculate denominator of z1 (Xjim)
                self.data_T_vals = get_item_array_from_subs(self.data_T, X.nonzero())
            else:
                self.data_T = np.zeros_like(X)
                self.data_T_vals = None
            self.X = preprocess(X)  # transform into sp tensor
        else:
            self.X = preprocess(X)  # transform into sp tensor
            if self.mutuality:
                # to use mutuality
                layer, i, j, m = self.X.subs

                # Transpose lijm -> ljim
                self.data_T = skt.sptensor(
                    subs=(layer, j, i, m), vals=self.X.vals.tolist(), shape=self.X.shape
                )
                # to calculate denominator of z1 (Xjim)
                self.data_T_vals = get_item_array_from_subs(self.data_T, self.X.subs).astype(int)

            else:
                self.data_T = skt.sptensor(
                    subs=tuple([np.array([], dtype="int8") for i in range(len(self.X.shape))]),
                    vals=[],
                    shape=self.X.shape,
                )
                self.data_T_vals = None

        self.subs_nz = self.X.subs
        self.sumX = self.X.vals.sum()

        self.L, self.N, self.M = X.shape[0], X.shape[1], X.shape[3]

        if "K" in extra_params:
            if extra_params["K"] is None:
                self.K = np.max(X.vals) + 1

                # logger.warning() vs warnings.warn() https://stackoverflow.com/a/14762106/843365
                msg = f"Parameter K was None. Defaulting to: {self.K}"
                warnings.warn(msg, UserWarning)
            else:
                self.K = int(extra_params["K"])
        else:

            if isinstance(X, skt.sptensor):
                self.K = X.vals.max() + 1
            else:
                self.K = int(X.max()) + 1

            # logger.warning() vs warnings.warn() https://stackoverflow.com/a/14762106/843365
            msg = f"Parameter K was None. Defaulting to: {self.K}"
            warnings.warn(msg, UserWarning)

        if "R" in extra_params:
            R = extra_params["R"]
            if R.shape != (self.L, self.N, self.N, self.M):
                msg = "Dimensions of reporter mask (R) do not match L x N x N x M"
                self.logger.error(msg)
                raise ValueError(msg)
        else:
            msg = "Reporters Mask was not informed (parameter R). "
            msg += "The model will assume that every reporter can report on any tie."

            # logger.warning() vs warnings.warn() https://stackoverflow.com/a/14762106/843365
            warnings.warn(msg, UserWarning)
            R = np.ones((self.L, self.N, self.N, self.M))
        self.R = preprocess(R)

        if "EPS" in extra_params:
            self.EPS = float(extra_params["EPS"])
        else:
            self.EPS = DEFAULT_EPS

        if "bias0" in extra_params:
            self.bias0 = float(extra_params["bias0"])
        else:
            self.bias0 = DEFAULT_BIAS0

        if "max_iter" in extra_params:
            self.max_iter = int(extra_params["max_iter"])
        else:
            self.max_iter = DEFAULT_MAX_ITER

        if "num_realisations" in extra_params:
            self.num_realisations = int(extra_params["num_realisations"])
        else:
            self.num_realisations = DEFAULT_NUM_REALISATIONS

        """
        HANDLE theta priors
        """
        if "alpha_theta" in extra_params or "beta_theta" in extra_params:

            self.alpha_theta = extra_params["alpha_theta"]
            self.beta_theta = extra_params["beta_theta"]

            if self.alpha_theta.shape != (self.L, self.M):
                msg = "alpha_theta matrix is not valid."
                msg += " When using this parameter, make sure to inform a %d x %d matrix."
                self.logger.error(msg)
                raise ValueError(msg % (self.L, self.M))

            if self.beta_theta.shape != (self.L, self.M):
                msg = "beta_theta matrix is not valid. When using this parameter, make sure to inform a %d x %d matrix."
                self.logger.error(msg)
                raise ValueError(msg % (self.L, self.M))

            warn_msg = "Ignoring theta_prior since full alpha_theta and beta_theta tensors were informed"
            self.logger.debug(warn_msg)

        else:
            if type(theta_prior) is not tuple or len(theta_prior) != 2:
                msg = "theta_prior must be a 2D tuple!"
                self.logger.error(msg)
                raise ValueError(msg)
            self.alpha_theta = theta_prior[0]
            self.beta_theta = theta_prior[1]

        """
        HANDLE lambda priors
        """
        if "alpha_lambda" in extra_params or "beta_lambda" in extra_params:

            self.alpha_lambda = extra_params["alpha_lambda"]
            self.beta_lambda = extra_params["beta_lambda"]

            if self.alpha_lambda.shape != (self.L, self.K):
                msg = "alpha_lambda matrix is not valid (dimensions = %d x %d)."
                msg += "When using this parameter, make sure to pass a %d x %d matrix."
                msg = msg % (self.alpha_lambda.shape[0], self.alpha_lambda.shape[1], self.L, self.K,)
                self.logger.error(msg)
                raise ValueError(msg)

            if self.beta_lambda.shape != (self.L, self.K):
                msg = "beta_lambda matrix is not valid (dimensions = %d x %d)."
                msg += "When using this parameter, make sure to pass a %d x %d matrix."
                msg = msg % (self.beta_lambda.shape[0], self.beta_lambda.shape[1], self.L, self.K,)
                self.logger.error(msg)
                raise ValueError(msg)

            warn_msg = "Ignoring lambda_prior since full alpha_lambda and beta_lambda tensors were informed"
            self.logger.debug(warn_msg)

        else:
            if type(lambda_prior) is not tuple or len(lambda_prior) != 2:
                msg = "lambda_prior must be a 2D tuple!"
                self.logger.error(msg)
                raise ValueError(msg)
            self.alpha_lambda = lambda_prior[0]
            self.beta_lambda = lambda_prior[1]

        if type(eta_prior) is not tuple or len(eta_prior) != 2:
            msg = "eta_prior must be a 2D tuple!"
            self.logger.error(msg)
            raise ValueError(msg)
        self.alpha_mutuality = eta_prior[0]
        self.beta_mutuality = eta_prior[1]

        if rho_prior is not None and rho_prior.shape != (self.L, self.N, self.N):
            msg = "rho_prior has to have shape equal to (L, N, N)!"
            self.logger.error(msg)
            raise ValueError(msg)
        self.rho_prior = rho_prior

        self._change_seed(seed)

    def fit(
        self,
        X: np.ndarray,
        theta_prior=(0.1, 0.1),
        lambda_prior=(10.0, 10.0),
        eta_prior=(0.5, 1.0),
        rho_prior=None,
        seed: int = None,
        **extra_params,
    ):
        """
        Parameters
        ----------
        X : ndarray
            Network adjacency tensor.
        theta_prior: 2D tuple
            Shape and scale hyperparameters for variable theta
        lambda_prior: 2D tuple
            Shape and scale hyperparameters for variable lambda
        eta_prior: 2D tuple
            Shape and scale hyperparameters for variable eta
        rho_prior : None/ndarray
            Array with prior values of the rho parameter - if ndarray.

        Extra parameters (Advanced tuning of inference)
        ----------------
        R: ndarray
            a multidimensional array L x N x N x M indicating which reports to consider
        K: None/int
            Value of the maximum entry of the network - i
        EPS : float
            White noise. Default: 1e-12
        bias0: float
            Bias for rho_prior entry 0. Default: 0.2
        max_iter: int
            Maximum number of iteration steps before aborting. Default=500

        Returns
        -------
        self.rho_f, self.G_exp_theta_f, self.G_exp_lambda_f, self.G_exp_nu_f, self.maxL
        """

        self.logger.debug("Checking user parameters passed to the VimureModel.fit()")
        self.__check_fit_params(
            X=X,
            theta_prior=theta_prior,
            lambda_prior=lambda_prior,
            rho_prior=rho_prior,
            eta_prior=eta_prior,
            seed=seed,
            **extra_params,
        )

        """
        Inference
        """
        maxL = -INF  # initialization of the maximum elbo
        trace = []  # Keep track of elbo values and running time

        for r in range(self.num_realisations):

            self.logger.debug("Initializing priors")
            if r < 5:  # the first 5 runs are with bias = 0
                self._set_rho_prior()
            else:
                curr_bias0 = (r - 4) * self.bias0
                self._set_rho_prior(bias0=curr_bias0)

            self._initialize_priors()
            self._initialize_old_variables()

            coincide = 0
            iter = 1
            reached_convergence = False

            elbo = -INF  # initialization of the elbo

            while not reached_convergence and iter <= self.max_iter:
                time_start = time.time()
                delta_gamma, delta_phi, delta_rho, delta_nu = self._update_CAVI(
                    data=self.X, subs_nz=self.subs_nz, data_T_vals=self.data_T_vals
                )
                runtime = time.time() - time_start

                iter, elbo, coincide, reached_convergence = self._check_for_convergence(
                    data=self.X,
                    data_T=self.data_T,
                    subs_nz=self.subs_nz,
                    r=r,
                    iter=iter,
                    elbo=elbo,
                    coincide=coincide,
                    reached_convergence=reached_convergence,
                )

                if (iter - 1) % 10 == 0:
                    trace.append((r, self.seed, iter - 1, elbo, runtime, reached_convergence))

            if maxL < elbo:
                self._update_optimal_parameters()
                maxL = elbo

            if self.seed is None:
                new_seed = self.prng.randint(1, 500)
            else:
                new_seed = self.seed + self.prng.randint(1, 500)

            self._change_seed(new_seed)
            # end cycle over realizations

        cols = ["realisation", "seed", "iter", "elbo", "runtime", "reached_convergence"]
        self.trace = pd.DataFrame(trace, columns=cols)

        self.maxL = maxL

        # TODO: Consider removing non-final copies of internal dataframes such as self.rho, self.gamma_rte
        #       and keep only the final versions (self.rho_f, self.gamma_rte) to save disk space
        #       when saving model to disk
        return self

    def _change_seed(self, seed):
        self.seed = seed
        self.prng = np.random.RandomState(seed)

    """
    INITIALISATION
    """

    def _set_rho_prior(self, bias0=DEFAULT_BIAS0):
        """
        Set prior on rho
        """

        # TODO: Allow self.rho_prior to be skt.sptensor if a user prefers sparse tensors.

        self.logger.debug("Setting priors on rho")
        # TODO: Make pr_rho sparse
        pr_rho = np.zeros((self.L, self.N, self.N, self.K))
        if self.rho_prior is None:
            pr_rho = 1 + 0.01 * self.prng.rand(self.L, self.N, self.N, self.K)

            if self.mutuality is True:
                pr_rho[:, :, :, 0] += bias0  # bias the 0-th entry to be higher
            else:
                pr_rho[:, :, :, 0] += bias0  # bias the 0-th entry to be higher

            if self.undirected:  # impose symmetry
                pr_rho = (pr_rho + np.transpose(pr_rho, axes=(0, 2, 1, 3))) / 2.0

            # Normalizing
            norm = pr_rho.sum(axis=-1)
            pr_rho /= norm[:, :, :, np.newaxis]
        else:

            sub_nz = self.rho_prior.nonzero()
            for k in range(self.K):
                pr_rho[(*sub_nz, k * np.ones(sub_nz[0].shape[0]).astype("int"))] = poisson.pmf(
                    k, self.rho_prior[sub_nz]
                ) + 1.0 * self.prng.rand(sub_nz[0].shape[0])

            if self.undirected:  # impose symmetry
                for layer in range(self.L):
                    for k in range(self.K):
                        pr_rho[layer, :, :, k] = (pr_rho[layer, :, :, k] + pr_rho[layer, :, :, k].T) / 2.0
            norm = pr_rho[sub_nz].sum(axis=-1)
            pr_rho[sub_nz] /= norm[:, np.newaxis]

        """
        INFORMATIVE PRIORS BASED ON REPORTERS' MASK & X UNION

        Calculate ties lij that have not been reported by anyone. R(lij,m) == 0 forall m
        # TODO: The code below is quite slow. Think of a faster way to do that without increasing memory footprint
        """
        # Step 1: Calculate all possible combinations of lij ties in format: dim (L*N*N, 3)
        all_possible_ties = (
            np.stack(np.meshgrid(range(self.L), range(self.N), range(self.N)))
            .reshape(-1, self.L * self.N * self.N)
            .T
        )
        all_possible_ties = pd.DataFrame(all_possible_ties, columns=["l", "i", "j"])

        def get_ties_not_reported(R_or_X):

            # Step 2: Calculate the union of lij ties in R (or X)
            if isinstance(R_or_X, skt.dtensor):
                subs_lij = R_or_X.nonzero()[0:3]
            else:
                subs_lij = R_or_X.subs[0:3]
            all_reported_ties = pd.DataFrame.from_dict({"l": subs_lij[0], "i": subs_lij[1], "j": subs_lij[2]})
            all_reported_ties.drop_duplicates(inplace=True)

            # Step 3: Which of the all_possible_ties do not appear in all_reported_ties?
            df = pd.merge(all_possible_ties, all_reported_ties, how="left", indicator=True)
            relevant_set = df[df["_merge"] == "left_only"][["l", "i", "j"]]

            return relevant_set

        ties_not_reported_by_R = get_ties_not_reported(self.R)

        if len(ties_not_reported_by_R) > 0:

            layer = ties_not_reported_by_R["l"]
            i = ties_not_reported_by_R["i"]
            j = ties_not_reported_by_R["j"]
            # Step 4: Prepare lij indices and set appropriate values in pr_rho(lij,k)
            pr_rho[layer, i, j, :] = 0
            pr_rho[layer, i, j, 0] = 1

        ties_not_reported_by_X = get_ties_not_reported(self.X)

        if len(ties_not_reported_by_X) > 0:

            layer = ties_not_reported_by_X["l"]
            i = ties_not_reported_by_X["i"]
            j = ties_not_reported_by_X["j"]
            # Step 4: Prepare lij indices and set appropriate values in pr_rho(lij,k)
            pr_rho[layer, i, j, :] = 0
            pr_rho[layer, i, j, 0] = 1

        self.pr_rho = pr_rho
        self.logpr_rho = np.log(self.pr_rho + self.EPS)

    def _initialize_priors(self):
        """
        Random initialization of the parameters theta, lambda, eta, rho.
        """

        self.logger.verbose("Setting priors for gamma_shp, phi_shp, gamma_rte, phi_rte")
        # TODO: Could these variables be made sparse?

        # we include some randomness
        self.gamma_shp = self.alpha_theta * self.prng.random_sample(size=(self.L, self.M)) + self.alpha_theta
        self.phi_shp = self.alpha_lambda * self.prng.random_sample(size=(self.L, self.K)) + self.alpha_lambda
        self.gamma_rte = self.beta_theta * self.prng.random_sample(size=(self.L, self.M)) + self.beta_theta
        self.phi_rte = self.beta_lambda * self.prng.random_sample(size=(self.L, self.K)) + self.beta_lambda

        self.logger.verbose("Setting priors for nu_shp, nu_rte")
        if self.mutuality:
            self.nu_shp = self.alpha_mutuality * self.prng.random_sample(1)[0] + self.alpha_mutuality
            self.nu_rte = self.beta_mutuality + self.sumX  # this is fixed once and for all
            self.G_exp_nu = np.exp(sp.psi(self.nu_shp) - np.log(self.nu_rte))
        else:  # not use the mutuality (eta ~ 0.)
            self.nu_shp = 0.000001
            self.nu_rte = 1.0
            self.G_exp_nu = 0.0

        self.rho = np.copy(self.pr_rho)

        self.G_exp_theta = np.exp(sp.psi(self.gamma_shp) - np.log(self.gamma_rte))
        self.G_exp_lambda = np.exp(sp.psi(self.phi_shp) - np.log(self.phi_rte))

    def _initialize_old_variables(self):
        """
        Initialize variables to keep the values of the parameters in the previous iteration.
        """

        self.gamma_shp_old = np.copy(self.gamma_shp)
        self.gamma_rte_old = np.copy(self.gamma_rte)
        self.phi_shp_old = np.copy(self.phi_shp)
        self.phi_rte_old = np.copy(self.phi_rte)
        self.rho_old = np.copy(self.rho)
        self.nu_shp_old = np.copy(self.nu_shp)

    """
    UPDATE VARIABLES
    """

    def _update_CAVI(self, data, subs_nz, data_T_vals=None):
        """
        Update parameters using Coordinate Ascent Variational Inference (CAVI)

        References:
            Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017).
            Variational Inference: A Review for Statisticians.
            Journal of the American Statistical Association, 112(518), 859–877.
            https://doi.org/10.1080/01621459.2017.1285773

        Parameters
        ----------
        data : sptensor/dtensor
            Network adjacency tensor.
        subs_nz : tuple
            Indices of elements of data that are non-zero.
        data_T_vals : ndarray/None
            Array with values of entries A[j, i] given non-zero entry (i, j) - if mutuality=True.
        """
        self.logger.verbose("Updating gamma")
        self._update_cache(data, subs_nz, data_T_vals)
        delta_gamma = self._update_gamma(subs_nz)

        self.logger.verbose("Updating phi")
        self._update_cache(data, subs_nz, data_T_vals)
        delta_phi = self._update_phi(subs_nz)

        self.logger.verbose("Updating rho")
        self._update_cache(data, subs_nz, data_T_vals)
        delta_rho = self._update_rho(subs_nz)
        if self.mutuality:
            self.logger.verbose("Updating nu")
            self._update_cache(data, subs_nz, data_T_vals)
            delta_nu = self._update_nu(subs_nz)
        else:
            delta_nu = 0.0

        return (delta_gamma, delta_phi, delta_rho, delta_nu)

    def _update_cache(self, data, subs_nz, data_T_vals=None):
        """
        Update the cache used in the CAVI update.

        Parameters
        ----------
        data : sptensor
            Network adjacency tensor.
        subs_nz : tuple
            Indices of elements of data that are non-zero.
        data_T_vals : ndarray/None
            Array with values of entries A[j, i] given non-zero entry (i, j) - if mutuality=True.
        """

        self.G_exp_theta = np.exp(sp.psi(self.gamma_shp) - np.log(self.gamma_rte))
        self.G_exp_lambda = np.exp(sp.psi(self.phi_shp) - np.log(self.phi_rte))

        if not self.mutuality:
            self.data_z1_nz = data.vals[:, np.newaxis].astype(float)
            self.data_z2_nz = None
        else:

            self.G_exp_nu = np.exp(sp.psi(self.nu_shp) - np.log(self.nu_rte))
            self.z1_nz = np.einsum(
                "I,Ik->Ik", self.G_exp_theta[subs_nz[0], subs_nz[3]], self.G_exp_lambda[subs_nz[0], :],
            )  # has dim= (I,K)
            self.z2_nz = self.G_exp_nu * data_T_vals  # has dim= (I)
            self.z_den_nz = self.z1_nz + self.z2_nz[:, np.newaxis]  # has dim= (I,K)
            self.z_den_nz[self.z_den_nz == 0] = 1
            self.data_z1_nz = data.vals[:, np.newaxis] * self.z1_nz / self.z_den_nz
            self.data_z2_nz = data.vals[:, np.newaxis] * self.z2_nz[:, np.newaxis] / self.z_den_nz

    def _update_gamma(self, subs_nz):

        self.gamma_shp = self.alpha_theta + self.sp_uttkrp_theta(self.data_z1_nz, subs_nz)

        E_phi_rho = np.einsum("lijk,lk->lij", self.rho, self.phi_shp / self.phi_rte)

        if isinstance(self.R, skt.dtensor):
            self.gamma_rte = self.beta_theta + np.einsum("lij,lijm->lm", E_phi_rho, np.array(self.R))
        else:

            self.gamma_rte = self.beta_theta * np.ones(shape=(self.L, self.M))
            # sum over k, final dim=I
            tmp = E_phi_rho[self.R.subs[0], self.R.subs[1], self.R.subs[2]]
            for c, (l, m) in enumerate(zip(*(self.R.subs[0], self.R.subs[3]))):  # sum over i,j
                self.gamma_rte[l, m] += tmp[c]

        dist_gs = np.amax(abs(self.gamma_shp - self.gamma_shp_old))
        dist_gr = np.amax(abs(self.gamma_rte - self.gamma_rte_old))
        dist_gamma = max(dist_gs, dist_gr)

        self.gamma_shp_old = np.copy(self.gamma_shp)
        self.gamma_rte_old = np.copy(self.gamma_rte)

        return dist_gamma

    def _update_phi(self, subs_nz):

        self.phi_shp = self.alpha_lambda + self.sp_uttkrp_lambda(self.data_z1_nz, subs_nz)

        out = np.zeros_like(self.phi_rte)

        if isinstance(self.R, skt.dtensor):
            subs = self.R.nonzero()
        else:
            subs = self.R.subs

        Egamma = self.gamma_shp[subs[0], subs[3]] / self.gamma_rte[subs[0], subs[3]]
        tmp = self.rho[subs[0], subs[1], subs[2], :] * Egamma[:, np.newaxis]  # dim is (I,K)
        for k in range(self.K):  # sum over i,j,m
            out[:, k] += np.bincount(subs[0], weights=tmp[:, k], minlength=self.L)

        self.phi_rte = self.beta_lambda + out

        # self.phi_rte = np.einsum('lijk,lm->lijmk', self.rho, self.gamma_shp / self.gamma_rte)
        # self.phi_rte = self.beta_lambda + np.einsum('lijmk,lijm->lk', self.phi_rte, self.R)

        dist_pa = np.amax(abs(self.phi_shp - self.phi_shp_old))
        dist_pr = np.amax(abs(self.phi_rte - self.phi_shp_old))
        dist_phi = max(dist_pa, dist_pr)

        self.phi_shp_old = np.copy(self.phi_shp)
        self.phi_rte_old = np.copy(self.phi_rte)

        return dist_phi

    def _update_rho(self, subs_nz):

        # TODO: Make Exp_theta_lambda sparse
        if isinstance(self.R, skt.dtensor):
            Exp_theta_lambda = np.einsum("lijm,lm->lij", np.array(self.R), self.gamma_shp / self.gamma_rte)
        else:

            tmp = pd.DataFrame.from_dict(
                {
                    "l": self.R.subs[0],
                    "i": self.R.subs[1],
                    "j": self.R.subs[2],
                    "m": self.R.subs[3],
                    "val": (self.gamma_shp / self.gamma_rte)[(self.R.subs[0], self.R.subs[3])] * self.R.vals,
                }
            )

            tmp = tmp.groupby(["l", "i", "j"])[["val"]].sum()
            tmp.reset_index(inplace=True)
            Exp_theta_lambda = skt.sptensor(
                subs=(tmp["l"], tmp["i"], tmp["j"]), vals=tmp["val"].tolist(), shape=(self.L, self.N, self.N)
            )
            Exp_theta_lambda = Exp_theta_lambda.toarray()

        self.logger.verbose("calculating einsum (Exp_theta_lambda vs phi_shp/phi_rte)")
        Exp_theta_lambda = np.einsum("lij,lk->lijk", Exp_theta_lambda, self.phi_shp / self.phi_rte)

        self.logger.verbose("calculating log_rho")
        log_rho = self.logpr_rho + self.sp_uttkrp_rho(self.data_z1_nz, subs_nz) - Exp_theta_lambda

        self.logger.verbose("finishing updating rho")
        self.rho = np.exp(log_rho)
        sums_over_k = self.rho.sum(axis=3)

        # normalize so that the sum over k is 1
        self.rho[sums_over_k > 0] /= sums_over_k[sums_over_k > 0, np.newaxis]

        dist_rho = np.amax(abs(self.rho - self.rho_old))

        self.rho_old = np.copy(self.rho)
        self.logger.verbose("finished updating rho")

        return dist_rho

    def _update_nu(self, subs_nz):

        self.nu_shp = (
            self.alpha_mutuality + (self.data_z2_nz * self.rho[subs_nz[0], subs_nz[1], subs_nz[2], :]).sum()
        )
        dist_nu = abs(self.nu_shp - self.nu_shp_old)

        self.nu_shp_old = np.copy(self.nu_shp)

        return dist_nu

    def sp_uttkrp_theta(self, vals, subs):
        """
        Compute the Khatri-Rao product (sparse version).

        Parameters
        ----------
        vals : ndarray
               Values of the non-zero entries.
        subs : tuple
               Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must
               be equal to the dimension of tensor.

        Returns
        -------
        out : ndarray
              Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao
              product of the membership matrix.
        """

        out = np.zeros_like(self.gamma_shp)

        # sum over k, final dim=I
        tmp = (self.rho[subs[0], subs[1], subs[2], :].astype(vals.dtype) * vals).sum(axis=1)
        for c, (l, m) in enumerate(zip(*(subs[0], subs[3]))):  # sum over i,j
            out[l, m] += tmp[c]
        return out

    def sp_uttkrp_lambda(self, vals, subs):
        """
        Compute the Khatri-Rao product (sparse version).

        Parameters
        ----------
        vals : ndarray
               Values of the non-zero entries.
        subs : tuple
               Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must
               be equal to the dimension of tensor.

        Returns
        -------
        out : ndarray
              Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao
              product of the membership matrix.
        """

        out = np.zeros_like(self.phi_shp)
        tmp = self.rho[subs[0], subs[1], subs[2], :].astype(vals.dtype) * vals  # dim is (I,K)
        for k in range(self.K):  # sum over i,j,m
            out[:, k] += np.bincount(subs[0], weights=tmp[:, k], minlength=self.L)

        return out

    def sp_uttkrp_rho(self, vals, subs):
        """
        Compute the Khatri-Rao product (sparse version).

        Parameters
        ----------
        vals : ndarray
               Values of the non-zero entries.
        subs : tuple
               Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must
               be equal to the dimension of tensor.

        Returns
        -------
        out : ndarray
              Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao
              product of the membership matrix.
        """

        out = np.zeros_like(self.rho)

        # TODO: try by removing ExpLog_theta
        ExpLog_theta = sp.psi(self.gamma_shp) - np.log(self.gamma_rte)
        ExpLog_lambda = sp.psi(self.phi_shp) - np.log(self.phi_rte)

        # dim is (I,K)
        # tmp = (ExpLog_lambda[subs[0], :]).astype(vals.dtype) * vals
        tmp = (ExpLog_theta[subs[0], subs[3]][:, np.newaxis] + ExpLog_lambda[subs[0], :]).astype(
            vals.dtype
        ) * vals
        # sum over m
        for c, (l, i, j) in enumerate(zip(*(subs[0], subs[1], subs[2]))):
            out[l, i, j, :] += tmp[c, :]

        return out

    def _update_optimal_parameters(self):
        """
        Update values of the parameters after convergence.
        """

        # Parameters
        self.gamma_shp_f = np.copy(self.gamma_shp)
        self.gamma_rte_f = np.copy(self.gamma_rte)
        self.phi_shp_f = np.copy(self.phi_shp)
        self.phi_rte_f = np.copy(self.phi_rte)
        self.nu_shp_f = np.copy(self.nu_shp)
        self.nu_rte_f = np.copy(self.nu_rte)
        self.rho_f = np.copy(self.rho)

        # Geometric expectations
        self.G_exp_theta_f = np.exp(sp.psi(self.gamma_shp_f) - np.log(self.gamma_rte_f))
        self.G_exp_lambda_f = np.exp(sp.psi(self.phi_shp_f) - np.log(self.phi_rte_f))
        self.G_exp_nu_f = np.exp(sp.psi(self.nu_shp_f) - np.log(self.nu_rte_f))

    """
    ELBO
    """

    def __ELBO(self, data, data_T, subs_nz):
        """
        Function to compute the ELBO.
        """

        E_exp_theta = self.gamma_shp / self.gamma_rte
        E_exp_lambda = self.phi_shp / self.phi_rte
        E_exp_rec = self.nu_shp / self.nu_rte

        E_PoissonMean = self.calculate_mean_poisson(
            G_exp_theta=E_exp_theta,
            G_exp_lambda=E_exp_lambda,
            G_exp_nu=E_exp_rec,
            rho=self.rho,
            X_T=data_T,
            R=self.R,
        )
        elbo = -E_PoissonMean.vals.sum()

        G_PoissonMean = self.calculate_mean_poisson(
            G_exp_theta=self.G_exp_theta,
            G_exp_lambda=self.G_exp_lambda,
            G_exp_nu=self.G_exp_nu,
            rho=np.exp(self.rho),
            X_T=data_T,
            R=self.R,
        )

        # Filter it to only include (l,i,j,m) that are in X
        rho_prior_lijm = pd.DataFrame.from_dict(
            {"l": subs_nz[0], "i": subs_nz[1], "j": subs_nz[2], "m": subs_nz[3]}
        )
        G_PoissonMean_lijm = pd.DataFrame.from_dict(
            {
                "l": G_PoissonMean.subs[0],
                "i": G_PoissonMean.subs[1],
                "j": G_PoissonMean.subs[2],
                "m": G_PoissonMean.subs[3],
                "val": G_PoissonMean.vals,
            }
        )

        logPoissonMean = pd.merge(rho_prior_lijm, G_PoissonMean_lijm, how="left")
        logPoissonMean.fillna(0, inplace=True)
        logPoissonMean = logPoissonMean["val"].values  # convert to numpy array

        logPoissonMean = np.log(logPoissonMean + self.EPS)
        elbo += (data.vals * logPoissonMean).sum()

        elbo += gamma_elbo_term(
            pa=self.alpha_theta, pb=self.beta_theta, qa=self.gamma_shp, qb=self.gamma_rte
        ).sum()
        elbo += gamma_elbo_term(
            pa=self.alpha_lambda, pb=self.beta_lambda, qa=self.phi_shp, qb=self.phi_rte
        ).sum()
        elbo += gamma_elbo_term(
            pa=self.alpha_mutuality, pb=self.beta_mutuality, qa=self.nu_shp, qb=self.nu_rte
        )

        elbo += categorical_elbo_term(self.rho, self.pr_rho, self.EPS).sum()

        if np.isnan(elbo):
            raise ValueError("ELBO is NaN!!!!")
            sys.exit(1)
        else:
            return elbo

    def _check_for_convergence(
        self,
        data: np.ndarray,
        data_T: np.ndarray,
        subs_nz,
        r: int,
        iter: int,
        elbo: float,
        coincide: int,
        reached_convergence,
    ):
        """
        Check for convergence by using the ELBO values.
        """

        if iter == 1 or iter % 10 == 0 or iter == self.max_iter:
            old_L = elbo
            # loglik = self.__Likelihood(data, data_T, subs_nz)
            elbo = self.__ELBO(data, data_T, subs_nz)

            if abs(elbo - old_L) < self.convergence_tol:
                coincide += 1
            else:
                coincide = 0

        if coincide > self.decision:
            reached_convergence = True

        if iter == 1 or iter % 10 == 0 or iter == self.max_iter:
            msg = f"Realisation {r:2} | Iter {iter:4} | ELBO value: {elbo:6.12f} | "
            msg += f"Reached convergence: {reached_convergence}"
            self.logger.debug(msg)

        iter += 1

        return iter, elbo, coincide, reached_convergence

    """
    UTILS
    """

    def calculate_mean_poisson(
        self, G_exp_theta=None, G_exp_lambda=None, G_exp_nu=None, rho=None, X_T=None, R=None,
    ):

        if G_exp_theta is None:
            G_exp_theta = self.G_exp_theta_f

        if G_exp_lambda is None:
            G_exp_lambda = self.G_exp_lambda_f

        if G_exp_nu is None:
            G_exp_nu = self.G_exp_nu_f

        if rho is None:
            rho = self.rho_f

        if X_T is None:
            X_T = self.data_T

        if R is None:
            R = self.R

        """
        exp value of: [theta * lambda + eta * X.T]
        """
        if isinstance(self.R, skt.dtensor):
            R_subs = self.R.nonzero()
        else:
            R_subs = self.R.subs

        # Following Equation 2:
        ThetaLambda = np.einsum("I,Ik->Ik", G_exp_theta[R_subs[0], R_subs[3]], G_exp_lambda[R_subs[0], :],)

        if isinstance(X_T, skt.dtensor) or isinstance(X_T, np.ndarray):
            PoissonMean = ThetaLambda + G_exp_nu * X_T[R_subs[0], R_subs[1], R_subs[2], R_subs[3], np.newaxis]
        else:
            R_lijm = pd.DataFrame.from_dict({"l": R_subs[0], "i": R_subs[1], "j": R_subs[2], "m": R_subs[3]})
            X_lijm = pd.DataFrame.from_dict(
                {"l": X_T.subs[0], "i": X_T.subs[1], "j": X_T.subs[2], "m": X_T.subs[3], "val": X_T.vals}
            )

            X_T_array = pd.merge(R_lijm, X_lijm, how="left")
            X_T_array.fillna(0, inplace=True)
            X_T_array = X_T_array["val"].values  # convert to numpy array

            PoissonMean = ThetaLambda + G_exp_nu * X_T_array[:, np.newaxis]

        vals = np.einsum("Ik,Ik->I", rho[R_subs[0], R_subs[1], R_subs[2], :], PoissonMean)
        rho_PoissonMean = skt.sptensor(subs=R_subs, vals=vals)

        return rho_PoissonMean


"""
UTIL FUNCTIONS FOR INFERENCE
"""


def gamma_elbo_term(pa, pb, qa, qb):
    return sp.gammaln(qa) - pa * np.log(qb) + (pa - qa) * sp.psi(qa) + qa * (1 - pb / qb)


def categorical_elbo_term(rho, prior_rho, EPS):
    K = rho.shape[-1]
    layer = np.zeros((rho.shape[0], rho.shape[1], rho.shape[2]))
    for k in range(K):
        layer += rho[:, :, :, k] * np.log(prior_rho[:, :, :, k] + EPS) - rho[:, :, :, k] * np.log(
            rho[:, :, :, k] + EPS
        )
    return layer
