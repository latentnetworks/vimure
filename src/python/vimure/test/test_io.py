import math
from multiprocessing.sharedctypes import Value
import os
import tempfile
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
        'r': ['Tom', 'Tom', 'jack', 'jack', 'Tom']
    }

    df = pd.DataFrame(data)
    return df

def synth_dataset_externals():
    """
    Produces a data frame with the expected column names
    """

    data = {
        'ego': ['Tom', 'nick', 'krish', 'jack', 'jack'],
        'alter': ['nick', 'Tom', 'jack', 'nick', 'nick'],
        'reporter': ['External Reporter 1', 
                     'External Reporter 1', 
                     'External Reporter 2', 
                     'External Reporter 1', 
                     'External Reporter 2']
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
            "Mapping used: ego='ego', alter='alter', reporter='r'. "
            "Hint: Use params ego,alter,... for mapping column names."
        )

        with pytest.raises(ValueError, match=expected_error_msg):
            vm.io.read_from_edgelist(synth_dataset_custom_names(), reporter="r")

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

    def test_reporters_parameter(self):
        """
        Tests that an error is raised when the reporters parameter is passed and
        is not a valid list.
        """ 
        error_msg = "'reporters' should be a list, instead it is of type: <class 'str'>."

        with pytest.raises(ValueError, match=error_msg):
            vm.io.read_from_edgelist(synth_dataset_standard_names(), 
                                     reporters="test") # type: ignore

        reporters_missing_jack = ['Tom', 'nick', 'krish']
        error_msg = (
            "Some reporters in the data frame do not appear "
            "in the list of reporters provided. "
            "Hint: Compare the unique values of the `reporter` column "
            "with the list of reporters passed as parameter."
        )
        with pytest.warns(None) as record:
            with pytest.raises(ValueError, match=error_msg):
                vm.io.read_from_edgelist(synth_dataset_standard_names(),
                                         reporters=reporters_missing_jack)
                    
                # Check that it registered the warning about nodes
                assert len(record) == 1
                
                first_warn_msg = (
                    "The set of nodes was not informed, "
                    "using ego and alter columns to infer nodes."
                )
                assert str(record[0].message) == first_warn_msg

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

            # Check nodes
            assert isinstance(net_obj.nodeNames, pd.DataFrame)
            assert net_obj.nodeNames.shape == (net_obj.N, 2)

            nodes = ['Tom', 'nick', 'krish', 'jack']
            assert net_obj.nodeNames['name'].tolist() == nodes

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

    def test_non_standard_names(self):
        """
        Tests that the function works with non standard column names
        if parameters are set up correctly.
        """

        df = synth_dataset_custom_names()

        with pytest.warns(None) as record:
            net_obj = vm.io.read_from_edgelist(df, ego="i", alter="j", reporter="r")

            assert net_obj.L == 1
            assert net_obj.K == 2
            assert net_obj.N == len(set(df['i'].tolist()+df['j'].tolist()))
            assert net_obj.M == net_obj.N

            # Check nodes
            assert isinstance(net_obj.nodeNames, pd.DataFrame)
            assert net_obj.nodeNames.shape == (net_obj.N, 2)

            nodes = ['Tom', 'nick', 'krish', 'jack']
            assert net_obj.nodeNames['name'].tolist() == nodes

        # Check that it registered several warnings
        assert len(record) == 4
        
        first_warn_msg = (
            "The set of nodes was not informed, "
            "using i and j columns to infer nodes."
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

    def test_standard_names_with_nodes_param(self):
        """
        Tests that the function works as expected when you pass the nodes parameter
        """

        df = synth_dataset_standard_names()
        nodes = ['Tom', 'nick', 'krish', 'jack']

        with pytest.warns(None) as record:
            net_obj = vm.io.read_from_edgelist(df, nodes=nodes, K=2)

            assert net_obj.L == 1
            assert net_obj.K == 2
            assert net_obj.N == len(set(df['ego'].tolist()+df['alter'].tolist()))
            assert net_obj.M == net_obj.N

            # Check nodes
            assert isinstance(net_obj.nodeNames, pd.DataFrame)
            assert net_obj.nodeNames.shape == (net_obj.N, 2)
            assert net_obj.nodeNames['name'].tolist() == nodes

            # Check that it registered several warnings
            assert len(record) == 2

            first_warn_msg =  (
                "The set of reporters was not informed, "
                "assuming set(reporters) = set(nodes) and N = M."
            )
            assert str(record[0].message) == first_warn_msg

            second_warn_msg =  (
                "Reporters Mask was not informed (parameter R). "
                "Parser will build it from reporter column, "
                "assuming a reporter can only report their own ties."
            )
            assert str(record[1].message) == second_warn_msg

    ### Tests edge cases

    def test_external_reporters_are_not_supported(self):
        """
        We do not support reporters that are not nodes.
        Test that it raises an error when this happens.
        """
        df = synth_dataset_externals()

        error_msg = (
            "This survey setup is not currently supported by the package: "
            " some reporters are not nodes in the network. "
            "Hint: If this is unexpected behaviour, "
            "compare the unique values of the `reporter` column "
            "with those of the `ego` and `alter` columns."
        )

        with pytest.warns(None) as record:
            with pytest.raises(ValueError, match=error_msg):
                vm.io.read_from_edgelist(df)

        # Check that it registered several warnings
        assert len(record) == 2
        
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

    def test_external_reporters_are_not_supported_even_if_informed(self):
        """
        We do not support reporters that are not nodes.
        Test that it raises an error when this happens.
        """
        df = synth_dataset_externals()
        reporters = ['External Reporter 1', 'External Reporter 2']

        error_msg = (
            "This survey setup is not currently supported by the package: "
            " some reporters are not nodes in the network. "
            "Hint: If this is unexpected behaviour, "
            "compare the unique values of the `reporter` column "
            "with those of the `ego` and `alter` columns."
        )

        with pytest.warns(None) as record:
            with pytest.raises(ValueError, match=error_msg):
                vm.io.read_from_edgelist(df, reporters=reporters)

        # Check that it registered several warnings
        assert len(record) == 1
        
        first_warn_msg = (
            "The set of nodes was not informed, "
            "using ego and alter columns to infer nodes."
        )
        assert str(record[0].message) == first_warn_msg

class TestReadFromCSV:

    def test_read_from_csv(self):
        """
        Tests that the function works as expected
        """
        
        df = synth_dataset_standard_names()

        # Save dataframe to a random file in the temp folder
        temp_filename = os.path.join(tempfile.gettempdir(), "test.csv")
        df.to_csv(temp_filename, index=False)
        print(temp_filename)

        with pytest.warns(None) as record:
            net_obj = vm.io.read_from_csv(temp_filename)

            assert net_obj.L == 1
            assert net_obj.K == 2
            assert net_obj.N == len(set(df['ego'].tolist()+df['alter'].tolist()))
            assert net_obj.M == net_obj.N

            # Check nodes
            assert isinstance(net_obj.nodeNames, pd.DataFrame)
            assert net_obj.nodeNames.shape == (net_obj.N, 2)

            nodes = ['Tom', 'nick', 'krish', 'jack']
            assert net_obj.nodeNames['name'].tolist() == nodes

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

    def test_read_from_csv_incorrect_name_mapping(self):
        """
        Tests that read_from_csv throws an error if the user
        does not provide the correct column names mapping.
        """

        df = synth_dataset_custom_names()

        # Save dataframe to a random file in the temp folder
        temp_filename = os.path.join(tempfile.gettempdir(), "test.csv")
        df.to_csv(temp_filename, index=False)
        print(temp_filename)

        expected_error_msg = (
            "Required columns not found in data frame: ego, alter, reporter. "
            "Mapping used: ego='ego', alter='alter', reporter='reporter'. "
            "Hint: Use params ego,alter,... for mapping column names."
        )

        with pytest.raises(ValueError, match=expected_error_msg):
            vm.io.read_from_csv(temp_filename)

    def test_non_standard_names_correct_mapping(self):
        df = synth_dataset_custom_names()

        # Save dataframe to a random file in the temp folder
        temp_filename = os.path.join(tempfile.gettempdir(), "test.csv")
        df.to_csv(temp_filename, index=False)
        print(temp_filename)

        with pytest.warns(None) as record:
            net_obj = vm.io.read_from_csv(temp_filename, 
                                          ego="i",
                                          alter="j",
                                          reporter="r")

            assert net_obj.L == 1
            assert net_obj.K == 2
            assert net_obj.N == len(set(df['i'].tolist()+df['j'].tolist()))
            assert net_obj.M == net_obj.N

            # Check nodes
            assert isinstance(net_obj.nodeNames, pd.DataFrame)
            assert net_obj.nodeNames.shape == (net_obj.N, 2)

            nodes = ['Tom', 'nick', 'krish', 'jack']
            assert net_obj.nodeNames['name'].tolist() == nodes

        # Check that it registered several warnings
        assert len(record) == 4
        
        first_warn_msg = (
            "The set of nodes was not informed, "
            "using i and j columns to infer nodes."
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
