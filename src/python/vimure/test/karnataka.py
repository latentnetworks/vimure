import os

import numpy as np
import pandas as pd
import vimure as vm

from vimure._log import setup_logging

logger = setup_logging("karnataka-script")

# Refer to the data folder within the tests directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_INPUT_FOLDER = os.path.join(CURRENT_DIR, "data/input/india_microfinance/formatted/")

LAYERS = ["advice", "kerorice", "money", "visit"]

FILE_PATTERN_EDGES = "%s_edges.csv"
FILE_PATTERN_META = "%s_meta.csv"

DEFAULT_PSEUDO_RANDOM_SEEDS = np.arange(10)


def read_village_data(
    village: str,
    ties_layer_mapping={
        "borrowmoney": "money",
        "receivemoney": "money",
        "receiveadvice": "advice",
        "helpdecision": "advice",
        "keroricego": "kerorice",
        "keroricecome": "kerorice",
        "visitgo": "visit",
        "visitcome": "visit",
    },
    data_folder: str = DATA_INPUT_FOLDER,
    print_details: bool = True,
    filter_layer: str = ""
):
    """
    IMPORTANT: Here we assume that only the people who actually appears as reporters
        in the edgelist CSV should be represented in the final network.

    There is a column 'didsurvey' in the metadata file but we (Elly and Jon) have opted
        not to use it for the time being because of some small inconsistencies in the original data leading to M < N.
    Our purpose here is to have a network containing only people who have actually answered questions about their ties.


    Parameters
    -------------

    village:
        ID of the village, as it is identified in the original CSV files
    accepted_ties:
        How ties encountered on these datasets should be mapped to layers in the final networks
    data_folder:
        Path to directory containg CSV files
    print_details:
        Whether to print out details about the dataset or not

    """

    edgelist = pd.read_csv(os.path.join(data_folder, FILE_PATTERN_EDGES % village))
    metadata = pd.read_csv(os.path.join(data_folder, FILE_PATTERN_META % village))

    if ties_layer_mapping is not None:
        # Convert ties to layers
        edgelist["layer"] = edgelist["type"].map(ties_layer_mapping)
    else:
        edgelist["layer"] = edgelist["type"]

    if filter_layer != "":
        edgelist = edgelist[edgelist["layer"] == filter_layer].copy()

    respondents = set(metadata[metadata["didsurv"] == 1]["pid"])
    nodes = respondents.union(set(edgelist["i"])).union(set(edgelist["j"]))

    # Remove self-loops
    edgelist = edgelist[edgelist["i"] != edgelist["j"]].copy()

    # Remove duplicates
    edgelist.drop_duplicates(inplace=True)
    edgelist.drop(columns=["type"], inplace=True)

    cols = ["reporter", "Ego", "Alter", "weight", "layer"]
    edgelist.columns = cols

    # Only keep reports made by those who were MARKED as respondents in metadata CSV
    edgelist = edgelist[edgelist["reporter"].isin(respondents)].copy()

    return edgelist, nodes, respondents


def parse_village(
    village: str, layers: list = LAYERS, data_folder: str = DATA_INPUT_FOLDER, filter_did_survey: bool = True,
):
    """
    Pre-process villages data from the structured CSV files (near match to raw data)

    Parameters
    -------------

    village:
        ID of the village, as it is identified in the original CSV files
    layers:
        List of layers that should be parsed
    data_folder:
        Path to directory containg CSV files
    filter_did_survey:
        True if we should restrict our datasets to people who took part in the survey

    """
    edgelist, nodes, respondents = read_village_data(village, data_folder=data_folder, print_details=False)

    return {
        layer: vm.io.parse_graph_from_edgelist(
            df=edgelist[edgelist["layer"] == layer].copy(),
            nodes=nodes,
            reporters=respondents,
            is_undirected=False,
        )
        for layer in layers
    }