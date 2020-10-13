#importing libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#loading the data

data_train = pd.read_csv("mini-competitionAAA/data/train.csv")
store = pd.read_csv("mini-competitionAAA/data/store.csv")

#getting rid of null values in the store column of the train set
data_train = data_train[~(data_train.loc[:, "Store"].isnull())]

# changing the store type to in before mergin with the store table
data_train.loc[:, "Store"] = data_train.loc[:, "Store"].astype('int64')

# merging the store table with the train table
data = pd.merge(data_train, store, how='left', on='Store')
