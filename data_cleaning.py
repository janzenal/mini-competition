import pandas as pd

def loading_data(data1, data2):
    #loading the data

    data1 = pd.read_csv("data/{}.csv".format(data1))
    data2 = pd.read_csv("data/{}.csv".format(data2))

    #getting rid of null values in the store column of the train set
    data1 = data1[~(data1.loc[:, "Store"].isnull())]

    # changing the store type to in before mergin with the stor table
    data1.loc[:, "Store"] = data1.loc[:, "Store"].astype('int64')

    # merging the store table with the train table
    data2 = pd.merge(data1, data2, how='left', on='Store')

    return data