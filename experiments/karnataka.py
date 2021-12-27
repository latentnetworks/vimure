import os
import inspect
import argparse
import time

import numpy as np
import pandas as pd
import vimure as vm

from vimure.log import setup_logging

logger = setup_logging("karnataka-script")

DATA_INPUT_FOLDER = "input/india_microfinance/formatted/"  # Folder contains structured CSV files

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

    respondents = set(metadata[metadata["didsurv"] == 1]["pid"])
    nodes = respondents.union(set(edgelist["i"])).union(set(edgelist["j"]))

    if ties_layer_mapping is not None:
        # Convert ties to layers
        edgelist["layer"] = edgelist["type"].map(ties_layer_mapping)
    else:
        edgelist["layer"] = edgelist["type"]

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


def main(
    village: str,
    data_folder: str,
    verbose=True,
    mutuality=True,
    seed=1,
    num_realisations=5,
    max_iter=101,
    calculate_baseline=False,
    calculate_reliability=True,
    calculate_reciprocity=True,
):
    """
    Given a selection of a network to run, run the model two times (once with mutuality=False and once with mutuality=True)
      with the same seed and return two Diagnostics object.

    NOTE: Parameters num_realisations and max_iter were selected based on a first demo trial with Village 1.
    In these tests, we noticed that ViMuRe reached convergence after around iteration 70.

    Parameters
    -----------

    networks: a BaseNetwork object representing a village+layer
        you can obtain this network object by running parse_village function
    verbose: bool
        TRUE would print a log of logs
    seed: int
        seed to use for the pseudorandom number generator
    num_realisations: int
        number of realisations of the Variational Inference to run
    max_iter: int
        stopping criteria for Coordinate Ascent Variational Inference (CAVI) optimisation algorithm.
        The algorithm can stop earlier if it reaches convergence.

    """

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    test_string = " ".join(["%s = %s" % (i, values[i]) for i in args])

    logger.info(f"Parsing | {test_string}")

    networks = parse_village(village, data_folder=os.path.join(data_folder, DATA_INPUT_FOLDER))

    for layer, network in networks.items():

        try:
            df_model_summary = pd.read_csv(f"{data_folder}/output/real_data/karnataka_villages/vimure_model_summary.csv")
            all_runs = list(map(tuple, df_model_summary[["village", "layer", "initial_seed"]].drop_duplicates().values))
            
            if (village, layer, seed) in all_runs:
                logger.info(f"Experiment {(village, layer, seed)} is already present in the output file - skipping it.")
                continue
            
        except Exception:
            pass
        
        """
        STEP 01: RUN MODEL
        """
        logger.info(f"Running layer {layer} | {test_string}")
        start_time = time.time()
        model = vm.model.VimureModel(mutuality=mutuality, undirected=False, verbose=verbose)
        model.fit(
            X=network.X, R=network.R, K=2, seed=seed, num_realisations=num_realisations, max_iter=max_iter,
        )
        end_time = time.time()
        running_time = end_time - start_time
        logger.info(f"Finished layer {layer} after {running_time} seconds | {test_string}")

        """
        STEP 02: SAVE df_model_details (See Notebook 04 - Single Run)
        """
        logger.info("Building DataFrame df_model_summary")
        df_model_summary = pd.DataFrame(
            {
                "running_time": running_time,
                "num_realisations": model.num_realisations,
                "max_iter": model.max_iter,
                "initial_seed": seed,
                "best_seed": model.seed,
                "best_elbo": model.maxL,
                "eta_est": model.G_exp_nu,
                "lambda_k": model.G_exp_lambda_f.tolist(),
                "model": "ViMuRe_T",
                "village": village,
                "layer": layer,
            },
            index=[0],
        )

        filename = f"{data_folder}/output/real_data/karnataka_villages/vimure_model_summary.csv"
        logger.info(f"Saving DataFrame at {filename}")
        print(filename)
        with open(filename, "a") as f:
            df_model_summary.to_csv(filename, mode="a", header=f.tell() == 0, index=False)

        """
        STEP 03: SAVE df_trace (See Notebook 04 - Single Run)
        """
        logger.info("Building DataFrame df_trace")
        df_trace = model.trace
        df_trace["model"] = "ViMuRe_T"
        df_trace["village"] = village
        df_trace["layer"] = layer

        filename = f"{data_folder}/output/real_data/karnataka_villages/vimure_model_trace.csv"
        logger.info(f"Saving DataFrame at {filename}")
        print(filename)
        with open(filename, "a") as f:
            df_trace.to_csv(filename, mode="a", header=f.tell() == 0, index=False)

        """
        STEP 04: SAVE df_edgelist (See Notebook 04 - Single Run)
        """
        logger.info("Calculating X_union & X_intersection & Y_vimure for comparison")

        sumX = np.sum(network.X.toarray(), axis=3)
        X_union = np.zeros(sumX.shape).astype("int")
        X_union[sumX > 0] = 1

        X_intersection = np.zeros(sumX.shape).astype("int")
        X_intersection[sumX == 2] = 1

        Y_vimure = vm.utils.apply_rho_threshold(model)

        logger.info("Building DataFrame df_edgelist")
        df_edgelist = pd.Series(
            {
                (ll, i, j): {
                    "village": village,
                    "layer": layer,
                    "initial_seed": seed,
                    "source": i,
                    "target": j,
                    "dyad_ID": f"{i}_{j}",
                    "source_report": np.array(network.X[ll, i, j, i]).flatten()[0] == 1,
                    "target_report": np.array(network.X[ll, i, j, j]).flatten()[0] == 1,
                    "vimure_posterior_probability": model.rho_f[ll, i, j, 1],
                    "in_union": X_union[ll, i, j] == 1,
                    "in_intersection": X_intersection[ll, i, j] == 1,
                    "in_vimure": Y_vimure[ll, i, j] == 1,
                    "reciprocated_in_union": X_union[ll, j, i] == 1,
                    "reciprocated_in_intersection": X_intersection[ll, j, i] == 1,
                    "reciprocated_in_vimure": Y_vimure[ll, j, i] == 1,
                }
                for ll in range(model.rho_f.shape[0])
                for i in range(model.rho_f.shape[1])
                for j in range(model.rho_f.shape[2])
            },
            name="stats",
        )
        df_edgelist = pd.DataFrame(df_edgelist.values.tolist())

        # Only keep dyads that were present in each of the baselines or inferred as True by ViMuRe
        valid_rows = df_edgelist["in_union"] | df_edgelist["in_intersection"] | df_edgelist["in_vimure"]
        df_edgelist = df_edgelist[valid_rows].copy()

        filename = f"{data_folder}/output/real_data/karnataka_villages/vimure_model_edgelist.csv"
        logger.info(f"Saving DataFrame at {filename}")
        print(filename)
        with open(filename, "a") as f:
            df_edgelist.to_csv(filename, mode="a", header=f.tell() == 0, index=False)

        """
        STEP 05: SAVE df_reliability (See Notebook 04)
        """
        logger.info("Building DataFrame df_reliability")
        reporters = set(network.R.subs[3])

        df_reliability = pd.Series(
            {
                (ll, m): {
                    "village": village,
                    "layer": layer,
                    "initial_seed": seed,
                    "node": m,
                    "theta": model.G_exp_theta_f[ll, m],
                    "lambda_theta": model.G_exp_lambda_f[ll, 1] * model.G_exp_theta_f[ll, m],
                    "is_node_reporter": m in reporters,
                }
                for ll in range(model.R.shape[0])
                for m in range(model.R.shape[1])  # Because N == M
            },
            name="stats",
        )
        df_reliability = pd.DataFrame(df_reliability.values.tolist())

        filename = f"{data_folder}/output/real_data/karnataka_villages/vimure_model_reliability.csv"
        logger.info(f"Saving DataFrame at {filename}")
        print(filename)
        with open(filename, "a") as f:
            df_reliability.to_csv(filename, mode="a", header=f.tell() == 0, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ViMuRe on Karnataka data")
    parser.add_argument("--village", type=str, help="the vilage to parse, read and run")

    parser.add_argument(
        "--data-folder",
        type=str,
        help="Where the data folder is located",
        dest="data_folder",
        default="/mnt/data",
    )

    args = parser.parse_args()

    for seed in DEFAULT_PSEUDO_RANDOM_SEEDS:
        main(
            village=args.village, seed=seed, data_folder=args.data_folder,
        )
