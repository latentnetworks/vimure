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
    sys.path.insert(0, "notebooks/python/experiments/")

    from karnataka import read_village_data # type: ignore
    df, nodes, reporters = read_village_data("vil1", 
                                             data_folder="data/input/india_microfinance/formatted/",
                                             filter_layer="money")
    df.rename(columns={"Ego": "ego", "Alter": "alter"}, inplace=True)
    yield df, nodes, reporters

