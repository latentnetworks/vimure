import os
import pandas as pd

ties_layer_mapping={
    "borrowmoney": "money",
    "lendmoney": "money",
    "giveadvice": "advice",
    "giveadvice": "advice",
    "keroricego": "kerorice",
    "keroricecome": "kerorice",
    "visitgo": "visit",
    "visitcome": "visit",
}

def get_karnataka_survey_data(village_id: int, tie_type: str,
                              indivinfo: pd.DataFrame,
                              ties_layer_mapping=ties_layer_mapping,
                              all_na_codes=["9999999", "5555555", "7777777", "0"],
                              raw_csv_folder="2010-0760_Data/Data/Raw_csv"):
    """
    Read the raw data for a given tie type and village id, 
    and return two dataframes: the edgelist and the list of respondents.

    Parameters
    ----------
    village_id : int
        The village id, between 1 and 10.
    tie_type : str
        The tie type
    indivinfo : pd.DataFrame
        The individual-level metadata
    all_na_codes : list, optional
        The list of codes that should be interpreted as missing values, by default ["9999999", "5555555", "7777777", "0"]
    raw_csv_folder : str, optional
        The path to the folder containing the raw csv files, by default "2010-0760_Data/Data/Raw_csv"

    Returns
    -------

    edgelist : pd.DataFrame
        The edgelist

    respondents : list
        The respondent metadata

    """

    # Filter the individual-level metadata to keep only the relevant village
    resp = indivinfo[indivinfo["village"] == village_id].copy()
    resp["didsurv"] = 1

    village_file = os.path.join(raw_csv_folder, f"village{village_id}.csv")
    metadata = pd.read_csv(village_file, header = None, names=["hhid", "ppid", "gender", "age"])

    ## gender (1-Male, 2-Female)
    metadata["gender"] = metadata["gender"].map({1: "Male", 2: "Female"})

    ## pre-process pid to match the format in the individual-level metadata
    metadata["ppid"] = metadata["ppid"].astype(str)
    metadata["hhid"] = metadata["hhid"].astype(str)
    metadata["pid"] =  metadata.apply(lambda x: f'{x["hhid"]}{0 if len(x["ppid"]) != 2 else ""}{x["ppid"]}', axis=1)

    ## Select only the relevant columns
    selected_cols = ["pid", "resp_status", "religion", "caste", "didsurv"]
    metadata = pd.merge(metadata, resp[selected_cols], on="pid", how="left")

    # Read the raw data
    filepath = os.path.join(raw_csv_folder, f"{tie_type}{village_id}.csv")
    df_raw = pd.read_csv(filepath, header=None, na_values=all_na_codes, dtype=str)

    # Example with the data
    edgelist = pd.melt(df_raw, id_vars=[0]).dropna()

    edgelist = edgelist.drop(columns="variable")\
          .rename(columns={0: "ego", "value": "alter"})\
          .assign(reporter=lambda x: x["ego"])

    # Let's also add a column for the tie type
    edgelist = edgelist.assign(tie_type=tie_type)

    # Let's add a weight column too
    edgelist = edgelist.assign(weight=1)

    # If the question was "Did you borrow money from anyone?", then we need to flip the ego and alter columns
    if tie_type in ["borrowmoney", "helpdecision", "keroricego", "visitgo"]:
        edgelist = edgelist.rename(columns={"ego": "alter", "alter": "ego"})

    edgelist["layer"] = edgelist["tie_type"].map(ties_layer_mapping)

    # Reorder the columns to make it easier to read
    edgelist = edgelist[["ego", "alter", "reporter", "tie_type", "layer", "weight"]]

    #### Further pre-processing steps ####

    # Who could actually report on the ties?
    reporters = set(metadata[metadata["didsurv"] == 1]["pid"])
    nodes = reporters.union(set(edgelist["ego"])).union(set(edgelist["alter"]))

    # Only keep reports made by those who were MARKED as reporters in metadata CSV
    edgelist = edgelist[edgelist["reporter"].isin(reporters)].copy()

    # Remove self-loops
    edgelist = edgelist[edgelist["ego"] != edgelist["alter"]].copy()

    # Remove duplicates
    edgelist.drop_duplicates(inplace=True)

    return edgelist, reporters

def get_layer(village_id, layer_name, indivinfo, 
              raw_csv_folder="2010-0760_Data/Data/Raw_csv"):

    tie_types = {
        "money": ["lendmoney", "borrowmoney"],
        "advice": ["giveadvice", "helpdecision"],
        "kerorice": ["keroricego", "keroricecome"],
        "visit": ["visitgo", "visitcome"],
    }

    selected_tie_types = tie_types[layer_name]

    edgelist = pd.DataFrame()
    reporters = set()

    for tie_type in selected_tie_types:
        edgelist_, reporters_ = get_karnataka_survey_data(village_id=village_id, 
                                                            tie_type=tie_type, 
                                                            indivinfo=indivinfo,
                                                            raw_csv_folder=raw_csv_folder)
        edgelist = pd.concat([edgelist, edgelist_])
        reporters = reporters.union(reporters_)

    return edgelist, reporters

DEFAULT_METADATA_FILEPATH = "datav4.0/Data/2. Demographics and Outcomes/individual_characteristics.dta" # nolint

def get_indivinfo(metadata_filepath=DEFAULT_METADATA_FILEPATH):
    indivinfo = pd.read_stata(metadata_filepath)
    indivinfo.drop_duplicates(subset=["pid"], inplace=True) ## one individual (6109803) is repeated twice.
    indivinfo["pid"] = indivinfo["pid"].astype(str)
    indivinfo["hhid"] = indivinfo["hhid"].astype(str)

    return indivinfo