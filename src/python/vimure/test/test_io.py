import math
from multiprocessing.sharedctypes import Value
import pytest
import logging

import numpy as np
import pandas as pd
import vimure as vm
import sktensor as skt

from sklearn.metrics import f1_score

logger = logging.getLogger("vm.test.test_model")
logger.setLevel(logging.DEBUG)

### Synthetic data

def synth_dataset_standard_names():
    """
    Produces a data frame with the expected column names
    """

    data = {
        'ego': ['Tom', 'nick', 'krish', 'jack', 'jack'],
        'alter': ['nick', 'Tom', 'jack', 'nick', 'nick'],
        'reporter': ['Tom', 'Tom', 'jack', 'jack', 'Tom']
    }

    df = pd.DataFrame(data)
    return df

def synth_dataset_custom_names():
    """
    Produces a data frame with the unexpected column names
    """

    data = {
        'i': ['Tom', 'nick', 'krish', 'jack', 'jack'],
        'j': ['nick', 'Tom', 'jack', 'nick', 'nick'],
        'respondent': ['Tom', 'Tom', 'jack', 'jack', 'Tom']
    }

    df = pd.DataFrame(data)
    return df

### Tests

class TestReadFromEdgelist:
    """
    Tests that vm.io functions works with the expected input types
    and raises errors when the input is not of the expected type.
    """

    ### Tests for error messages and warnings when input is not of the expected type

    def test_non_dataframe_input(self):
        """
        Tests that the function raises an error when the input is not a dataframe
        """

        error_msg = "'df' should be a DataFrame, instead it is of type: <class 'str'>."

        with pytest.raises(ValueError, match=error_msg):
            vm.io.read_from_edgelist("string") # type: ignore
            
    def test_no_required_columns(self):
        """
        Tests that an error is raised when the dataframe does not contain the expected columns
        and that the error message contains a hint on how to fix the problem.
        """

        expected_error_msg = (
            "Required columns not found in data frame: ego, alter, reporter. "
            "Mapping used: ego='ego', alter='alter', reporter='reporter'. "
            "Hint: Use params ego,alter,... for mapping column names."
        )

        with pytest.raises(ValueError, match=expected_error_msg):
            vm.io.read_from_edgelist(synth_dataset_custom_names())

    def test_only_ego_column(self):
        """
        Tests that an error is raised when the dataframe does not contain the expected columns
        and that the error message contains a hint on how to fix the problem.
        """

        expected_error_msg = (
            "Required columns not found in data frame: alter, reporter. "
            "Mapping used: ego='i', alter='alter', reporter='reporter'. "
            "Hint: Use params ego,alter,... for mapping column names."
        )

        with pytest.raises(ValueError, match=expected_error_msg):
            vm.io.read_from_edgelist(synth_dataset_custom_names(), ego="i")

    def test_only_alter_column(self):
        """
        Tests that an error is raised when the dataframe does not contain the expected columns
        and that the error message contains a hint on how to fix the problem.
        """

        expected_error_msg = (
            "Required columns not found in data frame: ego, reporter. "
            "Mapping used: ego='ego', alter='j', reporter='reporter'. "
            "Hint: Use params ego,alter,... for mapping column names."
        )

        with pytest.raises(ValueError, match=expected_error_msg):
            vm.io.read_from_edgelist(synth_dataset_custom_names(), alter="j")
 
    def test_only_reporter_column(self):
        """
        Tests that an error is raised when the dataframe does not contain the expected columns
        and that the error message contains a hint on how to fix the problem.
        """

        expected_error_msg = (
            "Required columns not found in data frame: ego, alter. "
            "Mapping used: ego='ego', alter='alter', reporter='respondent'. "
            "Hint: Use params ego,alter,... for mapping column names."
        )

        with pytest.raises(ValueError, match=expected_error_msg):
            vm.io.read_from_edgelist(synth_dataset_custom_names(), reporter="respondent")

    def test_nodes_parameter(self):
        """
        Tests that an error is raised when the nodes parameter is passed and
        is not a valid list.
        """ 
        error_msg = "'nodes' should be a list, instead it is of type: <class 'str'>."

        with pytest.raises(ValueError, match=error_msg):
            vm.io.read_from_edgelist(synth_dataset_standard_names(), nodes="test") # type: ignore

        nodes_missing_jack = ['Tom', 'nick', 'krish']
        error_msg = (
            "A list of nodes was informed, "
            "but it does not contain all nodes in the data frame."
        )
        with pytest.raises(ValueError, match=error_msg):
            vm.io.read_from_edgelist(synth_dataset_standard_names(),
                                     nodes=nodes_missing_jack)


    ### Tests for correct output

    def test_standard_names(self):
        """
        Tests that the function works with the expected column names
        """

        df = synth_dataset_standard_names()

        with pytest.warns(None) as record:
            net_obj = vm.io.read_from_edgelist(df)

            assert net_obj.L == 1
            assert net_obj.K == 2
            assert net_obj.N == len(set(df['ego'].tolist()+df['alter'].tolist()))
            assert net_obj.M == net_obj.N

        # Check that it registered several warnings
        assert len(record) == 4
        
        first_warn_msg = (
            "The set of nodes was not informed, "
            "using ego and alter columns to infer nodes."
        )
        assert str(record[0].message) == first_warn_msg

        second_warn_msg =  (
            "The set of reporters was not informed, "
            "assuming set(reporters) = set(nodes) and N = M."
        )
        assert str(record[1].message) == second_warn_msg

        third_warn_msg =  (
            "Reporters Mask was not informed (parameter R). "
            "Parser will build it from reporter column, "
            "assuming a reporter can only report their own ties."
        )
        assert str(record[2].message) == third_warn_msg

        fourth_warn_msg = "Parameter K was None. Defaulting to: 2"
        assert str(record[3].message) == fourth_warn_msg

    # def test_mapping_ego_and_alter(self):
    #     df = self.dataset()
    #     net_obj = vm.io.parse_graph_from_edgelist(df, ego="i", alter="j", reporter = "respondent")

    #     assert net_obj.L == 1
    #     assert net_obj.N == len(set(df['i'].tolist()+df['j'].tolist()))
    #     assert net_obj.M == net_obj.N

    #     model = vm.model.VimureModel(mutuality=True, verbose=True)
    #     model.fit(net_obj.X, R=net_obj.R, num_realisations=2, max_iter=21)


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
