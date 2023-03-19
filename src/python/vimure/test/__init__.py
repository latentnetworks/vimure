"""Unittests"""
import pytest

@pytest.fixture()
def karnataka_edgelist_vil1():

    import sys
    sys.path.insert(0, "notebooks/python/experiments/")

    from karnataka import read_village_data # type: ignore
    df, nodes, reporters = read_village_data("vil1", 
                                             data_folder="data/input/india_microfinance/formatted/",
                                             filter_layer="money")
    df.rename(columns={"Ego": "ego", "Alter": "alter"}, inplace=True)
    yield df, nodes, reporters