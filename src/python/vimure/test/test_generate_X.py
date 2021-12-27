import pytest

import numpy as np
import vimure as vm
import networkx as nx
import sktensor as skt

from copy import deepcopy


def check_objects(gt_network):

    """Check tensor X"""
    assert gt_network.X is not None, "X is None!"
    assert isinstance(gt_network.X, skt.sptensor) or isinstance(gt_network.X, skt.dtensor)
    assert gt_network.X.shape == (
        gt_network.L,
        gt_network.N,
        gt_network.N,
        gt_network.M,
    )
    assert np.sum(gt_network.X.vals) != 0.0, "Adjacency matrices of X are empty!"

    """Check theta"""
    assert gt_network.theta is not None, "theta is None!"
    assert type(gt_network.theta) == np.ndarray
    assert gt_network.theta.shape == (gt_network.L, gt_network.M)
    assert (gt_network.theta >= 0).all()

    """Check lambda_k"""
    assert gt_network.lambda_k is not None, "lambda_k is None!"
    assert type(gt_network.lambda_k) == np.ndarray
    assert gt_network.lambda_k.shape == gt_network.Y.shape
    assert (gt_network.lambda_k >= 0).all()
    assert gt_network.lambda_k.max() <= gt_network.K

    """Check R"""
    assert gt_network.R is not None, "R is None!"
    assert isinstance(gt_network.R, skt.sptensor) or isinstance(gt_network.R, skt.dtensor)
    assert gt_network.R.shape == (
        gt_network.L,
        gt_network.N,
        gt_network.N,
        gt_network.M,
    )

    if isinstance(gt_network.R, skt.sptensor):
        assert len(gt_network.R.vals) > 0, "Reporters' mask is full of zeros!"
    else:
        assert gt_network.R.sum() > skt.dtensor(0), "Reporters' mask is full of zeros!"


class TestGenerateXForStandardSBM:
    def test_generate_X_binarize_False(self):

        gt_network = vm.synthetic.StandardSBM(
            N=20,
            M=20,
            L=1,
            K=4,
            C=2,
            avg_degree=4,
            sparsify=False,
            seed=10,
        )

        gt_network.build_X(
            flag_self_reporter=False,
            cutoff_X=False,
            mutuality=0.5,
            seed=20,
        )

        check_objects(gt_network)

        msg = "X matrix matches Y perfectly, which is supposed to be very unlikely."
        for l in range(gt_network.X.shape[0]):
            for m in range(gt_network.X.shape[3]):
                np.testing.assert_raises(
                    AssertionError,
                    np.testing.assert_array_equal,
                    gt_network.Y.toarray()[l],
                    gt_network.X.toarray()[l, :, :m],
                )

    def test_generate_X_binarize_True(self):

        gt_network = vm.synthetic.StandardSBM(
            N=20, M=20, L=1, K=4, C=2, avg_degree=4, sparsify=False, seed=10
        )
        gt_network.build_X(flag_self_reporter=False, cutoff_X=True, mutuality=0.5, seed=20)

        check_objects(gt_network)
        assert np.max(gt_network.X.vals) <= (gt_network.K - 1)

        msg = "X matrix matches Y perfectly, which is supposed to be very unlikely."
        for l in range(gt_network.X.shape[0]):
            for m in range(gt_network.X.shape[3]):
                np.testing.assert_raises(
                    AssertionError,
                    np.testing.assert_array_equal,
                    gt_network.Y.toarray()[l],
                    gt_network.X.toarray()[l, :, :m],
                )

    def test_generate_X_different_seeds(self):
        """
        Tests whether running the function without informing any particular seed
            indeed leads to different tensors, guaranteeing that our code is not reusing seed parameters.
        """

        gt_network = vm.synthetic.StandardSBM(
            N=20, M=20, L=1, K=4, C=2, avg_degree=4, sparsify=False, seed=10
        )

        network1 = deepcopy(gt_network)
        network2 = deepcopy(gt_network)

        network1.build_X(flag_self_reporter=False, cutoff_X=False, mutuality=0.5, seed=18)
        network2.build_X(flag_self_reporter=False, cutoff_X=False, mutuality=0.5, seed=30)

        msg = "If assert fails, X1 matrix matches X2 perfectly -- which is supposed to be very unlikely."
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            network1.X.toarray(),
            network2.X.toarray(),
        )
