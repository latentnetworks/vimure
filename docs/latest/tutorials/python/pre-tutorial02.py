import sys
import pandas as pd

# Comment the two lines below if NOT running with ipython3
%load_ext autoreload
%autoreload 2

sys.path.insert(0, "docs/latest/tutorials/python/")
from tutorial01 import *;

METADATA_FILE = "docs/data/datav4.0/Data/2. Demographics and Outcomes/individual_characteristics.dta"
RAW_CSV_FOLDER = "docs/data/2010-0760_Data/Data/Raw_csv"

edgelist_money, reporters_money = get_layer(1, "money", get_indivinfo(METADATA_FILE), raw_csv_folder=RAW_CSV_FOLDER)
edgelist_visit, reporters_visit = get_layer(1, "visit", get_indivinfo(METADATA_FILE), raw_csv_folder=RAW_CSV_FOLDER)
edgelist_advice, reporters_advice = get_layer(1, "advice", get_indivinfo(METADATA_FILE), raw_csv_folder=RAW_CSV_FOLDER)
edgelist_kerorice, reporters_kerorice = get_layer(1, "kerorice", get_indivinfo(METADATA_FILE), raw_csv_folder=RAW_CSV_FOLDER)

edgelist = pd.concat([edgelist_money, edgelist_visit, edgelist_advice, edgelist_kerorice])
del edgelist_money, edgelist_visit, edgelist_advice, edgelist_kerorice
reporters = reporters_money.union(reporters_visit).union(reporters_advice).union(reporters_kerorice)
del reporters_money, reporters_visit, reporters_advice, reporters_kerorice

sys.path.insert(0, "src/python/")
import vimure as vm