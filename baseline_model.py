#importing libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from math import sqrt


#loading the data

data_train = pd.read_csv("data/train.csv")
store = pd.read_csv("data/store.csv")

#getting rid of null values in the store column of the train set
data_train = data_train[~(data_train.loc[:, "Store"].isnull())]

# changing the store type to in before mergin with the stor table
data_train.loc[:, "Store"] = data_train.loc[:, "Store"].astype('int64')

# mering the store table with the train table
data = pd.merge(data_train, store, how='left', on='Store')

# droping lots of columns
data = data.drop(columns=["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval", "Date", "StateHoliday", "SchoolHoliday", "Assortment", "StoreType"])

# dropping the rows with null values
data = data[~(data.loc[:, "DayOfWeek"].isnull()) &
            ~(data.loc[:, "Sales"].isnull()) &
            ~(data.loc[:, "Customers"].isnull()) &
            ~(data.loc[:, "Open"].isnull()) &
            ~(data.loc[:, "Promo"].isnull()) &
            ~(data.loc[:, "CompetitionDistance"].isnull()) &
            ~(data.loc[:, "Sales"] == 0.0)]

# splitting features and target back apart
x_train = data.copy(deep=True).drop(columns=["Sales"])
y_train = data.loc[:, "Sales"]


# defining evaluation metric
def compute_rmspe(actual, prediction):
    """
    Computes RMSE (root mean squared error) between predictions from a model
    and the actual values of the target variable.
    """

    rmspe = np.sqrt(np.mean(np.square(((actual - prediction) / actual)), axis=0))

    
    rmse = sqrt(mean_squared_error(actual, prediction))
    
    # rounding to 2 decimal places
    print('RMSE is ', round(rmse, 2))
    
    return rmse

lazy_estimator_predictions = pd.DataFrame(y_train.copy())

# using median of entire training set
lazy_estimator_predictions.loc[:, 'lazy_predicted_price'] = y_train.mean()
lazy_estimator_predictions.head().round()

lazy_estimator_rmspe = compute_rmspe(y_train, lazy_estimator_predictions.loc[:, 'lazy_predicted_sales'])

print("the predicte value of the baseline model (mean) is {}".format(lazy_estimator_predictions.iloc[2, 1]))
print("the RMSPE of the baseline model (mean) is {}".format(lazy_estimator_rmspe))