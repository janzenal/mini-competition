#importing libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


#loading the data

data_train = pd.read_csv("data/train.csv")
store = pd.read_csv("data/store.csv")

#getting rid of null values in the store column of the train set
data_train = data_train[~(data_train.loc[:, "Store"].isnull())]

# changing the store type to in before mergin with the stor table
data_train.loc[:, "Store"] = data_train.loc[:, "Store"].astype('int64')

# mering the store table with the train table
data = pd.merge(data_train, store, how='left', on='Store')


# transforming the StateHoliday column
data.loc[:, "StateHoliday"] = data.loc[:, "StateHoliday"].apply(lambda x: 1 if ((x == "a") or (x == "b")) else 0)
le = LabelEncoder()  #instantiate the Label Encoder

# label encoding the StoreType column
data.loc[:, 'StoreType'] = le.fit_transform(data.loc[:, 'StoreType'])

# label encoding the Assortment column
data.loc[:, 'Assortment'] = le.fit_transform(data.loc[:, 'Assortment'])

# dropping columns with high percentage of null values and useless values
data = data.drop(columns=["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval", "Date"])

# transforming the StateHoliday column
data.loc[:, "StateHoliday"] = data.loc[:, "StateHoliday"].apply(lambda x: 1 if ((x == "a") or (x == "b")) else 0)

# mean encoding the Store column
def mean_encode(data, col, on):
    group = data.groupby(col).mean()
    data.loc[:, col+'-original'] = data.loc[:, col]
    mapper = {k: v for k, v in zip(group.index, group.loc[:, on].values)}

    data.loc[:, col] = data.loc[:, col].replace(mapper)
    data.loc[:, col].fillna(value=np.mean(data.loc[:, col]), inplace=True)
    return data.loc[:, col]

columns_to_encode = ["Store"]

for col in columns_to_encode:
    data.loc[:, col] = mean_encode(data.copy(), col, "Sales")
    
# dropping the rows with null values
data = data[~(data.loc[:, "DayOfWeek"].isnull()) &
            ~(data.loc[:, "Sales"].isnull()) &
            ~(data.loc[:, "Customers"].isnull()) &
            ~(data.loc[:, "Open"].isnull()) &
            ~(data.loc[:, "Promo"].isnull()) &
            ~(data.loc[:, "SchoolHoliday"].isnull()) &
            ~(data.loc[:, "CompetitionDistance"].isnull()) &
            ~(data.loc[:, "Sales"] == 0.0)]

# splitting features and target back apart
x_train = data.copy(deep=True).drop(columns=["Sales"])
y_train = data.loc[:, "Sales"]

# defining evaluation metric
def compute_rmspe(actual, prediction):
    """
    Computs RMSPE (root mean squared percentage error) between predictions from a model
    and the actual values of the target variable.
    """

    rmspe = np.sqrt(np.mean(np.square(((actual - prediction) / actual)), axis=0))

    return rmspe

# linear regression
linear_regression_model = LinearRegression()
linear_regression_model.fit(x_train, y_train)
linear_regression_predictions = linear_regression_model.predict(x_train)

linear_regression_rmspe = compute_rmspe(y_train, linear_regression_predictions)
print("the RMSPE of the baseline model (mean) is {}".format(linear_regression_rmspe))