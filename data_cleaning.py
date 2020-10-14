import pandas as pd

# This section is for loading the data which will be applied for any model

def loading_data(data1, data2):
    
    # loading both tables, train and store
    if data1 == "test":
        data1 = pd.read_csv("data/{}.csv".format(data1), index_col=0)
    else:
        data1 = pd.read_csv("data/{}.csv".format(data1))
    data2 = pd.read_csv("data/{}.csv".format(data2))

    # getting rid of null values in the store column of the train set
    if data1.loc[:, "Store"].isnull().any():
        data1 = data1[~(data1.loc[:, "Store"].isnull())]

    # changing the store type to int before merging with the store table
    if data1.loc[:, "Store"].dtypes != 'int64':
        data1.loc[:, "Store"] = data1.loc[:, "Store"].astype('int64')

    # merging the store table with the train table
    data = pd.merge(data1, data2, how='left', on='Store')

    return data

#inspect the percentage of null values per column
def null_values(data):
    columns_with_nulls = []
    for column in data.columns:
        if data.loc[:, column].isnull().any():
            columns_with_nulls.append(column)

    for column in columns_with_nulls:
        percent_missing = round(((data.loc[data.loc[:, column].isnull()].shape[0] / data.shape[0]) * 100), 4)
        print("Column {} has {}% missing values \n".format(column, percent_missing))
