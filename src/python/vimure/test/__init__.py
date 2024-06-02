"""Unittests"""
import pytest

from os import devnull
from contextlib import contextmanager,redirect_stderr,redirect_stdout

### To guarantee that the experiments folder is in the path
from .karnataka import read_village_data

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
    
    df, nodes, reporters = read_village_data("vil1")
    df.rename(columns={"Ego": "ego", "Alter": "alter"}, inplace=True)
    yield df, nodes, reporters

@pytest.fixture()
def karnataka_edgelist_vil1_money():

    df, nodes, reporters = read_village_data("vil1", 
                                             filter_layer="money")
    df.rename(columns={"Ego": "ego", "Alter": "alter"}, inplace=True)
    yield df, nodes, reporters

