import math
import pytest
import logging

import numpy as np
import pandas as pd
import vimure as vm
import sktensor as skt

from sklearn.metrics import f1_score

logger = logging.getLogger("vm.test.test_model")
logger.setLevel(logging.DEBUG)


class TestIO:
    @pytest.mark.skip(
        reason="This is being reviewed. Checkout Notebook04 to see the current approach to reading this data."
    )
    def test_read_karnataka_village_from_csv(self):
        file = "/mnt/data/input/india_microfinance/formatted_singel_layer/vil57_visit.csv"
        net_obj = vm.io.parse_graph_from_csv(file)

        assert net_obj.L == 1
        assert net_obj.N == 214
        assert net_obj.M == 196

        assert isinstance(net_obj.X, skt.sptensor)
        assert net_obj.X.vals.sum() > 0  # Not all 0s
        assert net_obj.X.shape == (net_obj.L, net_obj.N, net_obj.N, net_obj.M)
        # https://stackoverflow.com/a/48648756/843365
        assert net_obj.X.vals.sum() < math.prod(net_obj.X.shape)  # Not all 1s

        assert isinstance(net_obj.R, skt.sptensor)
        assert net_obj.R.vals.sum() > 0  # Not all 0s
        assert net_obj.R.shape == (net_obj.L, net_obj.N, net_obj.N, net_obj.M)
        # https://stackoverflow.com/a/48648756/843365
        assert net_obj.R.vals.sum() < math.prod(net_obj.R.shape)  # Not all 1s

        """
        Since we didn't inform a custom reporters mask, reporters should only report their own ties
        """
        idxNodeFrom = 1
        idxNodeTo = 2
        idxM = 3

        msg = "Reporter %s is reporting some ties they are not involved in!"
        for m in range(net_obj.M):
            idx_reporting = np.argwhere(net_obj.R.subs[idxM] == m).flatten()

            nodes_from = net_obj.R.subs[idxNodeFrom][idx_reporting]
            nodes_to = net_obj.R.subs[idxNodeTo][idx_reporting]
            assert np.logical_or(nodes_from == m, nodes_to == m).all(), msg % m

    @pytest.mark.skip(
        reason="This is being reviewed. Checkout Notebook04 to see the current approach to reading this data."
    )
    def test_read_karnataka_village_multilayer(self):

        # Read all layers from village 57
        layers = ["visit", "kerorice", "money", "help"]
        pattern = "/mnt/data/input/india_microfinance/formatted_singel_layer/vil57_%s.csv"

        df = pd.concat([pd.read_csv(pattern % layer) for layer in layers])

        net_obj = vm.io.parse_graph_from_edgelist(df)

        assert net_obj.L == 4
        assert net_obj.N == 233
        assert net_obj.M == 229

        assert isinstance(net_obj.X, skt.sptensor)
        assert net_obj.X.vals.sum() > 0  # Not all 0s
        assert net_obj.X.shape == (net_obj.L, net_obj.N, net_obj.N, net_obj.M)
        # https://stackoverflow.com/a/48648756/843365
        assert net_obj.X.vals.sum() < math.prod(net_obj.X.shape)  # Not all 1s

        assert isinstance(net_obj.R, skt.sptensor)
        assert net_obj.R.vals.sum() > 0  # Not all 0s
        assert net_obj.R.shape == (net_obj.L, net_obj.N, net_obj.N, net_obj.M)
        # https://stackoverflow.com/a/48648756/843365
        assert net_obj.R.vals.sum() < math.prod(net_obj.R.shape)  # Not all 1s

        """
        Since we didn't inform a custom reporters mask, reporters should only report their own ties
        """
        idxNodeFrom = 1
        idxNodeTo = 2
        idxM = 3

        msg = "Reporter %s is reporting some ties they are not involved in!"
        for m in range(net_obj.M):
            idx_reporting = np.argwhere(net_obj.R.subs[idxM] == m).flatten()

            nodes_from = net_obj.R.subs[idxNodeFrom][idx_reporting]
            nodes_to = net_obj.R.subs[idxNodeTo][idx_reporting]
            assert np.logical_or(nodes_from == m, nodes_to == m).all(), msg % m
