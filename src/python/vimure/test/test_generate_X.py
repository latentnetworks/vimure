import pytest

import numpy as np
import pandas as pd
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
    def test_build_X_param_cutoff_X_False(self):

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

        gt_network._build_X(
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

    def test_build_X_param_cutoff_X_True(self):

        gt_network = vm.synthetic.StandardSBM(
            N=20, M=20, L=1, K=4, C=2, avg_degree=4, sparsify=False, seed=10
        )
        gt_network._build_X(flag_self_reporter=False, cutoff_X=True, mutuality=0.5, seed=20)

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

    def test_build_X_with_different_seeds(self):
        """
        Tests whether running the function without informing any particular seed
            indeed leads to different tensors, guaranteeing that our code is not reusing seed parameters.
        """

        gt_network = vm.synthetic.StandardSBM(
            N=20, M=20, L=1, K=4, C=2, avg_degree=4, sparsify=False, seed=10
        )

        network1 = deepcopy(gt_network)
        network2 = deepcopy(gt_network)

        network1._build_X(flag_self_reporter=False, cutoff_X=False, mutuality=0.5, seed=18)
        network2._build_X(flag_self_reporter=False, cutoff_X=False, mutuality=0.5, seed=30)

        msg = "If assert fails, X1 matrix matches X2 perfectly -- which is supposed to be very unlikely."
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            network1.X.toarray(),
            network2.X.toarray(),
        )

    def test_build_X_baselines_exist(self):
        """
        Asserts that Union and Intersection baselines are produced and are correct after build_X is invoked
        """

        gt_network = vm.synthetic.StandardSBM(
            N=20,
            M=20,
            L=1,
            K=2,
            C=2,
            avg_degree=4,
            sparsify=False,
            seed=10,
        )

        gt_network._build_X(
            flag_self_reporter=True,
            cutoff_X=True,
            mutuality=0.5,
            seed=20,
        )

        X = gt_network.X.toarray()

        """
        CHECK X_union
        """
        
        assert gt_network.X_union is not None, "X_union is None!"
        assert isinstance(gt_network.X_union, skt.sptensor) or isinstance(gt_network.X_union, skt.dtensor)
        assert gt_network.X_union.shape == (
            gt_network.L,
            gt_network.N,
            gt_network.N
        )

        # Calculate the expected X_union

        sumX = np.sum(X, axis=3)
        expected_X_union = np.zeros(sumX.shape).astype('int')
        expected_X_union[sumX > 0] = 1

        np.testing.assert_array_equal(gt_network.X_union.toarray(), 
                                      expected_X_union, 
                                      err_msg="X_union produced by the algorithm is not as expected")

        """
        CHECK X_intersection
        """
        assert gt_network.X_intersection is not None, "X_intersection is None!"
        assert (isinstance(gt_network.X_intersection, skt.sptensor) or 
                isinstance(gt_network.X_intersection, skt.dtensor))

        assert gt_network.X_intersection.shape == (
            gt_network.L,
            gt_network.N,
            gt_network.N,
        )

        expected_X_intersection = np.zeros(sumX.shape).astype('int')

        # Because flag_self_reporter=True, the maximum number of reporters that can report on a specific tie is 2. 
        # This is because, under this assumption, reporters are only asked about ties they are ego or alter.
        expected_X_intersection[sumX == gt_network.K] = 1

        np.testing.assert_array_equal(gt_network.X_intersection.toarray(), 
                                      expected_X_intersection, 
                                      err_msg="X_intersection produced by the algorithm is not as expected")

    def test_build_X_ensure_baselines_are_binary(self):
        """
        Asserts that Union and Intersection baselines are produced and correct even when K is higher than 2

        K represents the maximum entry value in the ground truth Y and it is also used by build_X to set Q 
        (if the user does not provide a value for Q).
        With this test, we want to ensure that the baselines are still binary regardless of K.

        """

        gt_network = vm.synthetic.StandardSBM(
            N=20,
            M=20,
            L=1,
            K=5, ## This is what makes this test different to the previous one
            C=2,
            avg_degree=4,
            sparsify=False,
            seed=10,
        )

        gt_network._build_X(
            flag_self_reporter=True,
            cutoff_X=True,
            mutuality=0.5,
            seed=20,
        )

        X = gt_network.X.toarray()

        """
        CHECK X_union
        """

        assert gt_network.X_union is not None, "X_union is None!"
        assert isinstance(gt_network.X_union, skt.sptensor) or isinstance(gt_network.X_union, skt.dtensor)
        assert gt_network.X_union.shape == (
            gt_network.L,
            gt_network.N,
            gt_network.N
        )

        # Calculate the expected X_union

        sumX = np.sum(X, axis=3)
        expected_X_union = np.zeros(sumX.shape).astype('int')
        expected_X_union[sumX > 0] = 1

        np.testing.assert_array_equal(gt_network.X_union.toarray(), 
                                        expected_X_union, 
                                        err_msg="X_union produced by the algorithm is not as expected")

        """
        CHECK X_intersection
        """
        assert gt_network.X_intersection is not None, "X_intersection is None!"
        assert (isinstance(gt_network.X_intersection, skt.sptensor) or 
                isinstance(gt_network.X_intersection, skt.dtensor))

        assert gt_network.X_intersection.shape == (gt_network.L, gt_network.N, gt_network.N)

        nonzero_lijm = pd.DataFrame(np.stack(gt_network.X.subs).T, columns=["l", "i", "j", "m"])

        # Because flag_self_reporter=True, the maximum number of reporters that can report on a specific tie is 2. 
        # This is because, under this assumption, reporters are only asked about ties they are ego or alter.
        nonzero_indices = nonzero_lijm.groupby(["l", "i", "j"]).count().query("m == 2")
        nonzero_indices = nonzero_indices.reset_index().drop(columns="m")

        expected_X_intersection = np.zeros(sumX.shape).astype('int')
        expected_X_intersection[nonzero_indices["l"], nonzero_indices["i"], nonzero_indices["j"]] = 1

        np.testing.assert_array_equal(gt_network.X_intersection.toarray(), 
                                        expected_X_intersection, 
                                        err_msg="X_intersection produced by the algorithm is not as expected")

