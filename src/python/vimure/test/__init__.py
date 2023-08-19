"""Unittests"""
import pytest

from os import devnull
from contextlib import contextmanager,redirect_stderr,redirect_stdout

@contextmanager
def suppress_stdout_stderr():
    """
    A context manager that redirects stdout and stderr to devnull
    Source: https://stackoverflow.com/a/52442331
    """
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


@pytest.fixture()
def karnataka_edgelist_vil1():

    import sys
    # Before inserting, we need to check if we're currently at <some-place>/vimure or at <some-place>/vimure/src/python
    current_path = sys.path[0]
    if current_path.endswith("vimure"):
        sys.path.insert(0, "notebooks/python/experiments/")
    elif current_path.endswith("src/python"):
        sys.path.insert(0, "../../notebooks/python/experiments/")
    else:
        raise Exception("Could not find the correct path to vimure")

    from karnataka import read_village_data # type: ignore
    df, nodes, reporters = read_village_data("vil1", 
                                             data_folder="data/input/india_microfinance/formatted/")
    df.rename(columns={"Ego": "ego", "Alter": "alter"}, inplace=True)
    yield df, nodes, reporters

@pytest.fixture()
def karnataka_edgelist_vil1_money():

    import os, sys;
    # Before inserting, we need to check if we're currently at <some-place>/vimure or at <some-place>/vimure/src/python
    current_path = os.getcwd()
    if current_path.endswith("vimure"):
        sys.path.insert(0, "notebooks/python/experiments/")
        data_folder = "data/input/india_microfinance/formatted/"
    elif current_path.endswith("python"):
        sys.path.insert(0, "../../notebooks/python/experiments/")
        data_folder = "../../data/input/india_microfinance/formatted/"
    else:
        raise Exception("Could not find the correct path to vimure")

    from karnataka import read_village_data # type: ignore
    df, nodes, reporters = read_village_data("vil1", 
                                             filter_layer="money",
                                             data_folder=data_folder)
    df.rename(columns={"Ego": "ego", "Alter": "alter"}, inplace=True)
    yield df, nodes, reporters

