import pytest
import logging
import numpy as np
import vimure as vm

from sklearn.metrics import f1_score

logger = logging.getLogger("vm.test.test_model")
logger.setLevel(logging.DEBUG)


class TestVimureModel:
    def check_final_parameters(self, model, synth_net):

        """Latent Variables - Parameter rho"""
        assert model.rho is not None
        assert type(model.rho) == np.ndarray
        assert model.rho.shape == (synth_net.L, synth_net.N, synth_net.N, synth_net.K)
        assert model.rho.sum() != 0, "Model returned a trivial solution"

        """Latent Variables - Parameter gamma - shape & rate"""
        assert model.gamma_shp is not None
        assert type(model.gamma_shp) == np.ndarray
        assert model.gamma_shp.shape == (synth_net.L, synth_net.M)
        assert model.gamma_shp.sum() > 0

        assert model.gamma_rte is not None
        assert type(model.gamma_rte) == np.ndarray
        assert model.gamma_rte.shape == (synth_net.L, synth_net.M)
        assert model.gamma_rte.sum() > 0

        """Latent Variables - Parameter phi - shape & rate"""
        assert model.phi_shp is not None
        assert type(model.phi_shp) == np.ndarray
        assert model.phi_shp.shape == (synth_net.L, synth_net.K)
        assert model.phi_shp.sum() > 0

        assert model.phi_rte is not None
        assert type(model.phi_rte) == np.ndarray
        assert model.phi_rte.shape == (synth_net.L, synth_net.K)
        assert model.phi_rte.sum() > 0

        """Latent Variables - Parameter nu - shape & rate"""
        assert model.nu_shp is not None
        assert type(model.nu_shp) == np.float64
        assert model.nu_shp >= 0

        assert model.nu_rte is not None
        assert type(model.nu_rte) == np.float64
        assert model.nu_rte >= 0

    def test_vimure_model_with_standard_sbm(self):
        logger.debug("Generating synthetic network (Standard SBM) for testing")
        gt_network = vm.synthetic.StandardSBM(
            N=20,
            M=20,
            L=1,
            K=3,
            C=2,
            avg_degree=2,
            sparsify=False,
        )

        logger.debug("Generating observed X from given Y")
        gt_network.build_X(flag_self_reporter=True)

        logger.debug("Starting VimureModel")
        model = vm.model.VimureModel()
        model.fit(gt_network.X, K=gt_network.K, R=gt_network.R)

        self.check_final_parameters(model, gt_network)

    def test_vimure_model_issues_appropriate_warnings(self):
        logger.debug("Generating synthetic network (Standard SBM) for testing")
        gt_network = vm.synthetic.StandardSBM(
            N=20,
            M=20,
            L=1,
            K=3,
            C=2,
            avg_degree=2,
            sparsify=False,
        )

        logger.debug("Generating observed X from given Y")
        gt_network.build_X(flag_self_reporter=True)

        logger.debug("Starting VimureModel without providing R")
        model = vm.model.VimureModel()
        with pytest.warns(UserWarning):
            model.fit(gt_network.X, K=gt_network.K)

        logger.debug("Starting VimureModel without providing K")
        model = vm.model.VimureModel()
        with pytest.warns(UserWarning):
            model.fit(gt_network.X, R=gt_network.R)

    def test_vimure_model_extreme_scenarios_over_reporting(self):
        """
        Ensures that the model achieves F1-score near 1.0 on a known
            extreme scenario (10% exaggerators over-reporting)
        """

        """
        GENERATE CUSTOM THETA
        """

        eta = 0.2
        theta_ratio = 0.1
        seed = 25
        K = 2

        gt_network = vm.synthetic.GMReciprocity(
            N=100,
            M=100,
            L=1,
            C=2,
            K=K,
            avg_degree=5,
            sparsify=True,
            seed=seed,  # seed=25 works well with default values
            ExpM=None,
            eta=eta,
        )

        theta = vm.synthetic.build_custom_theta(
            gt_network=gt_network,
            theta_ratio=theta_ratio,
            exaggeration_type="over",
            seed=seed,
        )

        LAMBDA_0 = 0.01
        LAMBDA_DIFF = 0.99
        gt_network.build_X(
            mutuality=eta,  # Same as planted mutuality
            theta=theta,  # Pre-defined theta
            cutoff_X=False,
            lambda_diff=LAMBDA_DIFF,
            flag_self_reporter=True,
            seed=seed,
            verbose=True,
        )

        lambda_k_GT = np.array([[LAMBDA_0, LAMBDA_0 + LAMBDA_DIFF]])
        beta_lambda = 10000 * np.ones((lambda_k_GT.shape))
        alpha_lambda = lambda_k_GT * beta_lambda

        model = vm.model.VimureModel(mutuality=True, verbose=True)
        model.fit(
            gt_network.X,
            K=K,
            seed=seed,
            theta_prior=(0.1, 0.1),
            eta_prior=(0.5, 1),
            alpha_lambda=alpha_lambda,
            beta_lambda=beta_lambda,
            num_realisations=2,
            max_iter=21,
            R=gt_network.R,
        )

        Y_true = gt_network.Y.toarray()[0].flatten()
        Y_rec = vm.utils.apply_rho_threshold(model, threshold=0.5)[0].flatten()

        assert np.allclose(f1_score(Y_true, Y_rec), 0.92, atol=1e-2)

    def test_vimure_model_extreme_scenarios_over_reporting_dense(self):
        """
        Ensures that the model achieves F1-score near 1.0 on a known
            extreme scenario (10% exaggerators over-reporting)
        """

        """
        GENERATE CUSTOM THETA
        """

        eta = 0.2
        theta_ratio = 0.1
        seed = 25
        K = 2

        gt_network = vm.synthetic.GMReciprocity(
            N=100,
            M=100,
            L=1,
            C=2,
            K=K,
            avg_degree=5,
            sparsify=True,
            seed=seed,  # seed=25 works well with default values
            ExpM=None,
            eta=eta,
        )

        theta = vm.synthetic.build_custom_theta(
            gt_network=gt_network,
            theta_ratio=theta_ratio,
            exaggeration_type="over",
            seed=seed,
        )

        LAMBDA_0 = 0.01
        LAMBDA_DIFF = 0.99
        gt_network.build_X(
            mutuality=eta,  # Same as planted mutuality
            theta=theta,  # Pre-defined theta
            cutoff_X=False,
            lambda_diff=LAMBDA_DIFF,
            flag_self_reporter=True,
            seed=seed,
            verbose=True,
        )

        lambda_k_GT = np.array([[LAMBDA_0, LAMBDA_0 + LAMBDA_DIFF]])
        beta_lambda = 10000 * np.ones((lambda_k_GT.shape))
        alpha_lambda = lambda_k_GT * beta_lambda

        model = vm.model.VimureModel(mutuality=True, verbose=True)
        model.fit(
            gt_network.X.toarray(),
            K=K,
            seed=seed,
            theta_prior=(0.1, 0.1),
            eta_prior=(0.5, 1),
            alpha_lambda=alpha_lambda,
            beta_lambda=beta_lambda,
            num_realisations=2,
            max_iter=21,
            R=gt_network.R.toarray(),
        )

        Y_true = gt_network.Y.toarray()[0].flatten()
        Y_rec = vm.utils.apply_rho_threshold(model, threshold=0.5)[0].flatten()

        assert np.allclose(f1_score(Y_true, Y_rec), 0.92, atol=1e-2)

    def test_vimure_model_extreme_scenarios_under_reporting(self):
        """
        Ensures that the model achieves F1-score near 1.0 on a known
            extreme scenario (10% exaggerators under-reporting)
        """

        """
        GENERATE CUSTOM THETA
        """

        eta = 0.2
        theta_ratio = 0.1
        seed = 25
        K = 2

        gt_network = vm.synthetic.GMReciprocity(
            N=100,
            M=100,
            L=1,
            C=2,
            K=K,
            avg_degree=5,
            sparsify=True,
            seed=seed,  # seed=25 works well with default values
            ExpM=None,
            eta=eta,
        )

        theta = vm.synthetic.build_custom_theta(
            gt_network=gt_network,
            theta_ratio=theta_ratio,
            exaggeration_type="under",
            seed=seed,
        )

        LAMBDA_0 = 0.01
        LAMBDA_DIFF = 0.99
        gt_network.build_X(
            mutuality=eta,  # Same as planted mutuality
            theta=theta,  # Pre-defined theta
            cutoff_X=False,
            lambda_diff=LAMBDA_DIFF,
            flag_self_reporter=True,
            seed=seed,
            verbose=True,
        )

        lambda_k_GT = np.array([[LAMBDA_0, LAMBDA_0 + LAMBDA_DIFF]])
        beta_lambda = 10000 * np.ones((lambda_k_GT.shape))
        alpha_lambda = lambda_k_GT * beta_lambda

        model = vm.model.VimureModel(mutuality=True, verbose=True)
        model.fit(
            gt_network.X,
            K=K,
            seed=seed,
            theta_prior=(0.1, 0.1),
            eta_prior=(0.5, 1),
            alpha_lambda=alpha_lambda,
            beta_lambda=beta_lambda,
            num_realisations=2,
            max_iter=21,
            R=gt_network.R,
        )

        Y_true = gt_network.Y.toarray()[0].flatten()
        Y_rec = vm.utils.apply_rho_threshold(model, threshold=0.5)[0].flatten()

        assert np.allclose(f1_score(Y_true, Y_rec), 0.97, atol=1e-2)
