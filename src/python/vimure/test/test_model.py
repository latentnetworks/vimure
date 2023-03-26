from typing import Any
import pytest
import logging

import numpy as np
import pandas as pd
import vimure as vm

from sklearn.metrics import f1_score

from . import karnataka_edgelist_vil1_money, suppress_stdout_stderr

logger = logging.getLogger("vm.test.test_model")

def check_final_parameters(model, net_obj):

    """Latent Variables - Parameter rho"""
    assert model.rho is not None
    assert type(model.rho) == np.ndarray
    assert model.rho.shape == (net_obj.L, net_obj.N, net_obj.N, net_obj.K)
    assert model.rho.sum() != 0, "Model returned a trivial solution"

    """Latent Variables - Parameter gamma - shape & rate"""
    assert model.gamma_shp is not None
    assert type(model.gamma_shp) == np.ndarray
    assert model.gamma_shp.shape == (net_obj.L, net_obj.M)
    assert model.gamma_shp.sum() > 0

    assert model.gamma_rte is not None
    assert type(model.gamma_rte) == np.ndarray
    assert model.gamma_rte.shape == (net_obj.L, net_obj.M)
    assert model.gamma_rte.sum() > 0

    """Latent Variables - Parameter phi - shape & rate"""
    assert model.phi_shp is not None
    assert type(model.phi_shp) == np.ndarray
    assert model.phi_shp.shape == (net_obj.L, net_obj.K)
    assert model.phi_shp.sum() > 0

    assert model.phi_rte is not None
    assert type(model.phi_rte) == np.ndarray
    assert model.phi_rte.shape == (net_obj.L, net_obj.K)
    assert model.phi_rte.sum() > 0

    """Latent Variables - Parameter nu - shape & rate"""
    assert model.nu_shp is not None
    assert type(model.nu_shp) == np.float64
    assert model.nu_shp >= 0

    assert model.nu_rte is not None
    assert type(model.nu_rte) == np.float64
    assert model.nu_rte >= 0

class TestVimureWithRandomNetworks:
    """
    Tests of the VimureModel class
    """

    def test_vimure_model_with_standard_sbm(self):
        """
        Tests the VimureModel class with a known synthetic network
        """

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
        gt_network._build_X(flag_self_reporter=True)

        logger.debug("Starting VimureModel")
        model = vm.model.VimureModel()
        with pytest.warns(None) as record:
            with suppress_stdout_stderr():
                model.fit(gt_network.X, K=gt_network.K, R=gt_network.R)

        check_final_parameters(model, gt_network)

    def test_vimure_model_issues_appropriate_warnings(self):
        """
        Tests the VimureModel class with a known synthetic network
        """

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
        gt_network._build_X(flag_self_reporter=True)

        logger.debug("Starting VimureModel without providing R")
        model = vm.model.VimureModel()
        with suppress_stdout_stderr():
            with pytest.warns(UserWarning):
                model.fit(gt_network.X, K=gt_network.K)

        logger.debug("Starting VimureModel without providing K")
        model = vm.model.VimureModel()
        with suppress_stdout_stderr():
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

        with pytest.warns(None) as record:
            gt_network = vm.synthetic.Multitensor(
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
            gt_network._build_X(
                mutuality=eta,  # Same as planted mutuality
                theta=theta,  # Pre-defined theta
                cutoff_X=False,
                lambda_diff=LAMBDA_DIFF,
                flag_self_reporter=True,
                seed=seed,
                verbose=False,
            )

        lambda_k_GT = np.array([[LAMBDA_0, LAMBDA_0 + LAMBDA_DIFF]])
        beta_lambda = 10000 * np.ones((lambda_k_GT.shape))
        alpha_lambda = lambda_k_GT * beta_lambda

        model = vm.model.VimureModel(mutuality=True, verbose=False)
        with pytest.warns(None) as record:
            with suppress_stdout_stderr():
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

        with pytest.warns(None) as record:
            gt_network = vm.synthetic.Multitensor(
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
            gt_network._build_X(
                mutuality=eta,  # Same as planted mutuality
                theta=theta,  # Pre-defined theta
                cutoff_X=False,
                lambda_diff=LAMBDA_DIFF,
                flag_self_reporter=True,
                seed=seed,
                verbose=False,
            )

        lambda_k_GT = np.array([[LAMBDA_0, LAMBDA_0 + LAMBDA_DIFF]])
        beta_lambda = 10000 * np.ones((lambda_k_GT.shape))
        alpha_lambda = lambda_k_GT * beta_lambda

        model = vm.model.VimureModel(mutuality=True, verbose=False)
        with pytest.warns(None) as record:
            with suppress_stdout_stderr():
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

        with pytest.warns(None) as record:
            gt_network = vm.synthetic.Multitensor(
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
            gt_network._build_X(
                mutuality=eta,  # Same as planted mutuality
                theta=theta,  # Pre-defined theta
                cutoff_X=False,
                lambda_diff=LAMBDA_DIFF,
                flag_self_reporter=True,
                seed=seed,
                verbose=False,
            )

        lambda_k_GT = np.array([[LAMBDA_0, LAMBDA_0 + LAMBDA_DIFF]])
        beta_lambda = 10000 * np.ones((lambda_k_GT.shape))
        alpha_lambda = lambda_k_GT * beta_lambda

        model = vm.model.VimureModel(mutuality=True, verbose=False)
        with pytest.warns(None) as record:
            with suppress_stdout_stderr():
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


class TestVimureWithReadData:
    """
    Tests the model with real data
    """

    def test_inform_nodes_reporters(self, karnataka_edgelist_vil1_money):
        df, nodes, reporters = karnataka_edgelist_vil1_money

        with pytest.warns(None) as record:
            net_obj = vm._io.read_from_edgelist(df, K=2, nodes=list(nodes), reporters=list(reporters))

        model = vm.model.VimureModel(mutuality=True, verbose=False)
        with suppress_stdout_stderr():
            model.fit(
                net_obj.X,
                K=2,
                seed=1,
                theta_prior=(0.1, 0.1),
                eta_prior=(0.5, 1),
                num_realisations=2,
                max_iter=21,
                R=net_obj.R,
            )


class TestInferredModel:
    @classmethod
    def setup_class(cls):
        cls.synth_net = vm.synthetic.StandardSBM(K=2)
        cls.synth_net._build_X(flag_self_reporter=True)

        cls.model = vm.model.VimureModel()
        with suppress_stdout_stderr():
            cls.model.fit(
                cls.synth_net.X,
                K=cls.synth_net.K,
                R=cls.synth_net.R,
                num_realisations=1,
                max_iter=500,
            )

        cls.model_mutuality = vm.model.VimureModel(mutuality=True)
        cls.model_mutuality.fit(
            cls.synth_net.X,
            K=cls.synth_net.K,
            R=cls.synth_net.R,
            num_realisations=1,
            max_iter=500,
        )

    def check_output(self, Y, model):
        assert Y.shape == (model.L, model.N, model.N)
        assert Y.sum() > 0

    def test_non_implemented_method(self):
        with pytest.raises(ValueError) as e_info:
            self.model.get_inferred_model(method="NotImplemented")
            assert (
                str(e_info.value)
                == '\'method\' should be one of "rho_max", "rho_mean","fixed_threshold", "heuristic_threshold".'
            )

    def test_fixed_threshold(self):
        with pytest.raises(ValueError) as e_info:
            self.model.get_inferred_model(method="fixed_threshold")
            assert (
                str(e_info.value)
                == 'For method="fixed_threshold", you must set the threshold to a value in [0,1].'
            )

        with pytest.raises(ValueError) as e_info:
            self.model.get_inferred_model(method="fixed_threshold", threshold=2)
            assert (
                str(e_info.value)
                == 'For method="fixed_threshold", you must set the threshold to a value in [0,1].'
            )

        Y = self.model.get_inferred_model(method="fixed_threshold", threshold=0.5)
        self.check_output(Y, self.model_mutuality)

    def test_rho_max(self):
        Y = self.model_mutuality.get_inferred_model(method="rho_max")
        self.check_output(Y, self.model_mutuality)

        Y = self.model.get_inferred_model(method="rho_max")
        self.check_output(Y, self.model)

    def test_rho_mean(self):
        Y = self.model_mutuality.get_inferred_model(method="rho_mean")
        self.check_output(Y, self.model_mutuality)

        Y = self.model.get_inferred_model(method="rho_mean")
        self.check_output(Y, self.model)

    def test_heuristic_threshold(self):
        Y = self.model_mutuality.get_inferred_model(
            method="heuristic_threshold"
        )
        self.check_output(Y, self.model_mutuality)

        Y = self.model.get_inferred_model(method="heuristic_threshold")
        self.check_output(Y, self.model)
    
    def test_sample_Y(self):
        N = 10
        Y = self.model.sample_inferred_model(N=N)
        
        assert len(Y) == N
        
        for y in Y:
            self.check_output(y, self.model_mutuality)


class TestVimureRealData:

    def test_internal_api(self, karnataka_edgelist_vil1_money):
    
        df, _, _ = karnataka_edgelist_vil1_money

        with pytest.warns(None) as record:
            net_obj = vm._io.read_from_edgelist(df, K=2)

        model = vm.model.VimureModel()

        with suppress_stdout_stderr():
            model.fit(net_obj.X, K=net_obj.K, R=net_obj.R)

        check_final_parameters(model, net_obj)

    def test_data_as_edgelist(self, karnataka_edgelist_vil1_money):
    
        df, _, _ = karnataka_edgelist_vil1_money

        # Run model directly with a pandas dataframe
        model = vm.model.VimureModel()
        with pytest.warns(None) as record:
            with suppress_stdout_stderr():
                model.fit(df, 
                        seed=1,
                        num_realisations=1,
                        max_iter=500)

        # Read in data as an edgelist to compare
        with pytest.warns(None) as record:
            net_obj = vm._io.read_from_edgelist(df, K=2)
        check_final_parameters(model, net_obj)