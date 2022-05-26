import pytest

import numpy as np
import vimure as vm
import networkx as nx
import sktensor as skt

from vimure.synthetic import (
    DEFAULT_N,
    DEFAULT_M,
    DEFAULT_K,
    DEFAULT_L,
    DEFAULT_C,
    DEFAULT_EXP_IN,
    DEFAULT_EXP_OUT,
    DEFAULT_AVG_DEGREE,
    DEFAULT_SEED,
    DEFAULT_ETA,
)


class TestBaseSyntheticNetwork:
    def check_objects(self, synth_net):
        assert synth_net.C == DEFAULT_C
        assert synth_net.N == DEFAULT_N
        assert synth_net.M == DEFAULT_M
        assert synth_net.L == DEFAULT_L
        assert synth_net.avg_degree == DEFAULT_AVG_DEGREE
        assert synth_net.Y.shape == (DEFAULT_L, DEFAULT_N, DEFAULT_N)

        assert isinstance(synth_net.Y, skt.sptensor) or isinstance(synth_net.Y, skt.dtensor)
        assert synth_net.Y.vals.sum() != 0.0, "Adjacency matrices are empty!"
        assert synth_net.Y.vals.max() < synth_net.K


class TestStandardSBM(TestBaseSyntheticNetwork):
    @classmethod
    def setup_class(cls):
        """
        Creates a single StandardSBM instance object to be shared by tests in this class
        """
        cls.synth_net = vm.synthetic.StandardSBM(
            N=DEFAULT_N,
            M=DEFAULT_M,
            L=DEFAULT_L,
            K=DEFAULT_K,
            C=DEFAULT_C,
            avg_degree=DEFAULT_AVG_DEGREE,
            sparsify=False,
        )

        # TODO: Add specific test cases for networks with overlapping communities
        cls.overlapping_net = vm.synthetic.StandardSBM(
            N=DEFAULT_N,
            M=DEFAULT_M,
            L=DEFAULT_L,
            K=DEFAULT_K,
            C=DEFAULT_C,
            avg_degree=DEFAULT_AVG_DEGREE,
            overlapping=0.1,
            normalization=False,
            corr=0.2,
        )

        cls.sparse_synth_net = vm.synthetic.StandardSBM(
            N=DEFAULT_N,
            M=DEFAULT_M,
            L=DEFAULT_L,
            K=DEFAULT_K,
            C=DEFAULT_C,
            avg_degree=DEFAULT_AVG_DEGREE,
            sparsify=True,
        )

    def test_yields_appropriate_objects(self):
        """
        CHECK matrices and objects have the right types and shapes and values
        """

        self.check_objects(self.synth_net)
        self.check_objects(self.sparse_synth_net)

    def test_yields_appropriate_affinity_matrix(self):
        """
        Focusing on matrix w: did it yield expected values?
        """

        for l in range(self.synth_net.L):

            msg = "Dimmensions of affinity matrix w[l=%d] are incorrect." % l
            assert self.synth_net.w[l].shape == (
                DEFAULT_C,
                DEFAULT_C,
            ), msg

            diag_vals = self.synth_net.w[l].diagonal()

            non_diag_indices = (
                np.ones(shape=self.synth_net.w[l].shape) - np.identity(len(self.synth_net.w[l]))
            ).astype(bool)
            non_diag_vals = self.synth_net.w[l][non_diag_indices]

            msg = "Diagonal values in the affinity matrix should be %s than non-diag."

            # TODO: Remove iteration over C
            for c in enumerate(range(self.synth_net.C)):
                diag_arr, non_diag_arr = np.meshgrid(diag_vals, non_diag_vals)
                if self.synth_net.structure[l] == "assortative":
                    assert np.alltrue(diag_arr > non_diag_arr), msg % "greater"
                else:
                    assert np.alltrue(diag_arr < non_diag_arr), msg % "smaller"


class TestDegreeCorrectedSBM(TestStandardSBM):
    @classmethod
    def setup_class(cls):
        """
        Creates a single StandardSBM instance object to be shared by tests in this class
        """
        cls.synth_net = vm.synthetic.DegreeCorrectedSBM(
            N=DEFAULT_N,
            M=DEFAULT_M,
            L=DEFAULT_L,
            K=DEFAULT_K,
            avg_degree=DEFAULT_AVG_DEGREE,
            C=DEFAULT_C,
            sparsify=False,
        )

        cls.sparse_synth_net = vm.synthetic.DegreeCorrectedSBM(
            N=DEFAULT_N,
            M=DEFAULT_M,
            L=DEFAULT_L,
            K=DEFAULT_K,
            C=DEFAULT_C,
            avg_degree=DEFAULT_AVG_DEGREE,
            sparsify=True,
        )

    def test_yields_appropriate_objects(self):
        """
        CHECK matrices and objects have the right types and shapes and values
        """

        super().test_yields_appropriate_objects()

        assert self.synth_net.exp_in == DEFAULT_EXP_IN
        assert self.synth_net.exp_out == DEFAULT_EXP_OUT

        assert len(self.synth_net.d_in) == self.synth_net.N
        assert len(self.synth_net.d_out) == self.synth_net.N

    def test_degree_correction_was_indeed_applied(self):
        """
        Does the Y matrix also change when we change exp_in and exp_out?
        """
        another_synth_net = vm.synthetic.DegreeCorrectedSBM(
            N=DEFAULT_N,
            M=DEFAULT_M,
            L=DEFAULT_L,
            K=DEFAULT_K,
            C=DEFAULT_C,
            avg_degree=DEFAULT_AVG_DEGREE,
            exp_in=4,
            exp_out=6,
            sparsify=False,
        )
        another_sparse_synth_net = vm.synthetic.DegreeCorrectedSBM(
            N=DEFAULT_N,
            M=DEFAULT_M,
            L=DEFAULT_L,
            K=DEFAULT_K,
            C=DEFAULT_C,
            avg_degree=DEFAULT_AVG_DEGREE,
            exp_in=4,
            exp_out=6,
            sparsify=True,
        )

        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            self.synth_net.Y.toarray(),
            another_synth_net.Y.toarray(),
        )
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            self.sparse_synth_net.Y.toarray(),
            another_sparse_synth_net.Y.toarray(),
        )

        assert np.sum(self.synth_net.Y.vals) > np.sum(
            self.sparse_synth_net.Y.vals
        ), "self.synth_net.Y should be denser than self.sparser_synth_net.Y"
        assert np.sum(self.synth_net.Y.vals) > np.sum(
            another_synth_net.Y.vals
        ), "another_synth_net should be denser than self.synth_net.Y"
        assert np.sum(another_synth_net.Y.vals) > np.sum(
            another_sparse_synth_net.Y.vals
        ), "another_sparser_synth_net.Y should be denser than self.sparser_synth_net.Y"


class TestMultitensor(TestBaseSyntheticNetwork):
    @classmethod
    def setup_class(cls):
        """
        Creates a single Multitensor instance object to be shared by tests in this class
        """
        cls.synth_net = vm.synthetic.Multitensor(
            N=DEFAULT_N,
            M=DEFAULT_M,
            K=DEFAULT_K,
            L=DEFAULT_L,
            C=DEFAULT_C,
            eta=DEFAULT_ETA,
            avg_degree=DEFAULT_AVG_DEGREE,
        )

    def test_yields_appropriate_objects(self):
        """
        CHECK matrices and objects have the right types and shapes and values
        """

        self.check_objects(self.synth_net)

        assert self.synth_net.eta == DEFAULT_ETA
        assert self.synth_net.ExpM is not None
