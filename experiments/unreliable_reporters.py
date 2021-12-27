"""
Runs an experiment, with multiple combinations of parameters, considering two major extreme scenarios:
    i)  one in which a portion of reporters are perfectly reliable but some (theta_ratio) tend to under-report ties
    ii) one in which a portion of reporters are perfectly reliable but some (theta_ratio) tend to over-report ties

To run it, open a terminal and type:

cd latent_network_models/code/vimure
docker-compose run --rm notebooks python3 experiments/unreliable_reporters.py
"""

# Temporarily importing the library with sys.path until we can't install it with pip install vimure
import time

import numpy as np
import scipy as sp
import pandas as pd
import vimure as vm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, f1_score, mean_squared_error
from sklearn.model_selection import GridSearchCV


DEFAULT_THETA_RATIO_VALS = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
]

# Guarantees we have 10 realisations for every experiment
DEFAULT_PSEUDO_RANDOM_SEEDS = np.arange(10)

DEFAULT_ETA = 0.0  # Without mutuality


class UnreliableReportersExp(RegressorMixin, BaseEstimator):
    """
    This class replicates a scikit-learn structure to represent an experiment.

    By extending from sklearn, we can make better use of Grid Search and the library's metrics such as f1-score

    """

    def __init__(
        self,
        gt_network: vm.synthetic.BaseSyntheticNetwork,
        eta: float = DEFAULT_ETA,
        theta_ratio: float = None,
        exaggeration_type: str = None,
        mutuality: bool = None,
        seed: int = None,
        verbose: bool = True,
        num_realisations=10,
        max_iter=21,
    ):

        self.gt_network = gt_network
        self.K = gt_network.K

        self.eta = eta
        self.seed = seed
        self.mutuality = mutuality
        self.exaggeration_type = exaggeration_type
        self.theta_ratio = theta_ratio

        self.num_realisations = num_realisations
        self.max_iter = max_iter
        self.verbose = verbose

        self.logger = vm.log.setup_logging("experiment.UnreliableReportersExp", verbose)

    def fit(self, X, y):
        """

        Parameters
        ------------
        X: np.ndarray or pandas Data.Frame
            (Not used)

            This parameter is there just to please scikit-learn's grid search format of class structure.
            To make it work seamlessly with other features of scikit-learn, pass a X matrix containing the same number
            of rows as there are in the y parameter.

        y: np.ndarray or pandas Series or list

            The ground truth

        """

        """
        GENERATE CUSTOM THETA
        """

        self.logger.info(
            "Building custom_theta with parameters:"
            f" theta_ratio={np.round(self.theta_ratio, 2)} &"
            f" exaggeration_type={self.exaggeration_type}"
        )
        theta = vm.synthetic.build_custom_theta(
            gt_network=self.gt_network,
            theta_ratio=self.theta_ratio,
            exaggeration_type=self.exaggeration_type,
            seed=self.seed,
        )

        """
        GENERATE X
        """
        msg = "Generating X with same eta as ground truth: eta=%.2f" % self.eta
        self.logger.info(msg)
        LAMBDA_0 = 0.01
        LAMBDA_DIFF = 0.99
        self.gt_network.build_X(
            mutuality=self.eta,
            theta=theta,  # Pre-defined theta
            cutoff_X=False,
            lambda_diff=LAMBDA_DIFF,
            flag_self_reporter=True,
            seed=self.seed,
            verbose=self.verbose,
        )

        self.X = self.gt_network.X
        self.R = self.gt_network.R
        self.theta = self.gt_network.theta
        self.lambda_k = self.gt_network.lambda_k

        """
        Y_union baseline
        """
        sumX = np.sum(self.X.toarray(), axis=3)
        Y_union = np.zeros(sumX.shape).astype("int")
        Y_union[sumX > 0] = 1
        self.Y_union = Y_union

        """
        Y_intersection baseline
        """
        Y_intersection = np.zeros(sumX.shape).astype("int")
        Y_intersection[sumX == 2] = 1
        self.Y_intersection = Y_intersection

        """
        RUN INFERENCE MODEL
        """
        with_or_without = {True: "WITH", False: "WITHOUT"}
        msg = "Running ViMuRe model %s mutuality..." % with_or_without[self.mutuality]
        self.logger.info(msg)

        """
        PRIORS
        """

        # Replicate single layer ground truth lambda_k
        lambda_k_GT = np.array([[LAMBDA_0, LAMBDA_0 + LAMBDA_DIFF]])
        beta_lambda = 10000 * np.ones((lambda_k_GT.shape))
        alpha_lambda = lambda_k_GT * beta_lambda
        theta_prior = (0.1, 0.1)

        time_start = time.time()
        model = vm.model.VimureModel(mutuality=self.mutuality, verbose=self.verbose)
        model.fit(
            self.X,
            K=self.K,
            seed=self.seed,  # initial seed
            theta_prior=theta_prior,
            eta_prior=(0.5, 1),
            alpha_lambda=alpha_lambda,
            beta_lambda=beta_lambda,
            num_realisations=self.num_realisations,
            max_iter=self.max_iter,
            R=self.R,
            bias0=0.2,
        )
        msg = f"ViMuRe model took {np.round(time.time() - time_start, 2)} seconds to run'"
        self.logger.info(msg)

        self.model = model

    def predict(self, X):
        """
        Returns the Y as estimated by the model
        """

        rho_sub = self.model.rho_f[0, :, :, 1]
        fixed_threshold = 0.01

        rho_sub[rho_sub < fixed_threshold] = 0
        rho_sub[rho_sub >= fixed_threshold] = 1

        return rho_sub.flatten()


def main(
    available_seeds: list = DEFAULT_PSEUDO_RANDOM_SEEDS,
    theta_ratio_vals: list = DEFAULT_THETA_RATIO_VALS,
    exaggeration_type: list = ["under", "over"],
    mutuality: list = [True, False],
    reciprocity_Y: float = 0.2,
    eta: float = DEFAULT_ETA,
    n_jobs=-1,
    gt_network_seed=25,
    save_output=True,
    verbose=True,
    **kwargs,
):
    """
    Runs an experiment, with multiple combinations of parameters, considering two major extreme scenarios:
        i)  one in which a portion of reporters are perfectly reliable but some (theta_ratio) tend to under-report ties
        ii) one in which a portion of reporters are perfectly reliable but some (theta_ratio) tend to over-report ties

    Parameters
    -----------

    available_seeds: list of integers
        The length of this list controls the number of realisations you will get for every scenario tested

    theta_ratio_vals: list of floats
        All possible theta_ratio to be tested in under- and over-reporting scenarios

    selected_eta: float
        Mutuality parameter to be used on ground truth network as well as the inference model

    n_jobs: int

        Number of jobs to run in parallel.
        None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
        If running inside Docker, this means all processors available to the Docker container.
        See https://scikit-learn.org/stable/glossary.html#term-n_jobs for more details.

    """

    module_logger = vm.log.setup_logging("experiments.unreliable", verbose)

    if "N" in kwargs:
        N = kwargs["N"]
    else:
        N = 100

    if "M" in kwargs:
        M = kwargs["M"]
    else:
        M = N

    if "L" in kwargs:
        if kwargs["L"] > 1:
            msg = "Invalid L. This experiment only supports single-layer currently."
            raise ValueError(msg)
    L = 1

    if "C" in kwargs:
        C = kwargs["C"]
    else:
        C = 2

    if "K" in kwargs:
        K = kwargs["K"]
    else:
        K = 2

    if "avg_degree" in kwargs:
        avg_degree = kwargs["avg_degree"]
    else:
        avg_degree = 5

    if "num_realisations" in kwargs:
        num_realisations = int(kwargs["num_realisations"])
    else:
        num_realisations = 10

    if "max_iter" in kwargs:
        max_iter = int(kwargs["max_iter"])
    else:
        max_iter = 21

    """
    SETUP GROUND TRUTH

    Create a Ground Truth Network, common to all experiments.
    """

    module_logger.info("Creating ground truth network with eta=%.2f..." % eta)

    time_start = time.time()

    # GMReciprocity won't always produce a single giant component, selecting one seed that is guaranteed to produce one.
    gt_network = vm.synthetic.GMReciprocity(
        N=N,
        M=M,
        L=L,
        C=C,
        K=K,
        eta=reciprocity_Y,
        ExpM=None,
        avg_degree=avg_degree,
        sparsify=True,
        seed=gt_network_seed,
    )
    msg = f"GMReciprocity took {np.round(time.time() - time_start, 2)} seconds to run'"
    module_logger.info(msg)

    """
    SETUP EXPERIMENTS

    Explore all possible parameters. If using default, this means 10 x 11 x 2 x 2 = 440 combinations
    """

    # All experiments will use the same ground truth network
    experiment_base = UnreliableReportersExp(
        gt_network, num_realisations=num_realisations, max_iter=max_iter, verbose=verbose, eta=eta,
    )

    parameters = {
        "seed": available_seeds,
        "theta_ratio": theta_ratio_vals,
        "exaggeration_type": exaggeration_type,
        "mutuality": mutuality,
    }

    # Uncomment this section for a shorter DEBUG run
    #     parameters = {
    #         "seed": [0],
    #         "theta_ratio": [0, 0.2],
    #         "exaggeration_type": ["under", "over"],
    #         "mutuality": [True, False],
    #     }
    # n_jobs = 1

    """
    GRID SEARCH

    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    Here we are using a trick to perform grid search without cross-validation by setting cv=[(slice(None), slice(None)]
        since we want to be able to emulate the inference on the full data.
    https://stackoverflow.com/a/44682305/843365

    """

    # GridSearchCV treats all problems as classification problems, so we need to provide the ground truth beforehand
    Y_flatten = gt_network.Y.toarray()[0].flatten()

    # GridSearchCV also requires that we provide a matrix of features X (as it is common in ML-settings).
    # Because what we are doing is a different thing altogether, simulation, this matrix X will not be used by
    #   grid search at all.
    # Therefore, let us just create a sparse fake_X to have it work with our code.
    fake_X = sp.sparse.dok_matrix((len(Y_flatten), 1))

    scoring = {
        "mse": make_scorer(mean_squared_error, greater_is_better=False),
        "f1": make_scorer(f1_score, greater_is_better=True),
    }

    time_start = time.time()
    clf = GridSearchCV(
        estimator=experiment_base,
        param_grid=parameters,
        cv=[(slice(None), slice(None))],
        scoring=scoring,
        refit="f1",
        verbose=True,
        n_jobs=n_jobs,  # How many parallel jobs to run, tune accordingly
        return_train_score=True,
    )

    clf.fit(fake_X, Y_flatten)

    clf.eta = eta
    msg = f"Grid search is complete after {np.round((time.time() - time_start)/60, 2)} minutes"
    module_logger.info(msg)

    results_df = pd.DataFrame(clf.cv_results_)
    cols = results_df.columns
    selected_cols = cols[["param_" in col for col in cols]].tolist()
    selected_cols.extend(["mean_test_f1", "mean_test_mse", "std_test_f1", "std_test_mse", "mean_fit_time"])

    results_df = results_df[selected_cols]
    results_df["param_theta_ratio"] = results_df["param_theta_ratio"].astype(float)
    results_df["param_seed"] = results_df["param_seed"].astype(str)
    results_df["eta"] = eta

    if save_output:
        print(results_df)
        filename = "/mnt/data/exp_unreliable_reporters.csv"
        module_logger.info("Saving results to %s" % filename)
        results_df.to_csv(filename, index=False)

    return clf, results_df


if __name__ == "__main__":
    main()
