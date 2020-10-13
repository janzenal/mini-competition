#importing libraries
import pandas as pd
import numpy as np
from math import sqrt

#dropping columns and rows with null values
def delete_nulls(data):
    data = data.drop(columns=["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                              "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval", 
                              "Date", "StateHoliday", "SchoolHoliday", 
                              "Assortment", "StoreType"])
    data = data[~(data.loc[:, "DayOfWeek"].isnull()) &
                ~(data.loc[:, "Sales"].isnull()) &
                ~(data.loc[:, "Customers"].isnull()) &
                ~(data.loc[:, "Open"].isnull()) &
                ~(data.loc[:, "Promo"].isnull()) &
                ~(data.loc[:, "CompetitionDistance"].isnull()) &
                ~(data.loc[:, "Sales"] == 0.0)]
    return data

# splitting features and target back apart
def split(data):
    x_train = data.copy(deep=True).drop(columns=["Sales"])
    y_train = data.loc[:, "Sales"]
    return x_train, y_train

# using the mean of the entire training set as a first prediction
def lazy_estimator(target):
    lazy_estimator_predictions = pd.DataFrame(target.copy())
    lazy_estimator_predictions.loc[:, 'lazy_predicted_price'] = target.mean()
    predict = lazy_estimator_predictions.loc[:, 'lazy_predicted_price']
    return predict

# defining evaluation metric
def compute_rmspe(actual, prediction):
    rmspe = np.sqrt(np.mean(np.square(((actual - prediction) / actual)), axis=0)) 
    return rmspe

# our baseline model using the functions defined above
def baseline(data):
    # deleting rows with null values and getting rid of some columns
    cleaned_data = delete_nulls(data)
    
    # splitting the data into features and target
    X, y = split(cleaned_data)
    
    # calculating the prediction which is simply the mean here
    predict = lazy_estimator(y)
    
    # computing the RMSPE of the difference between the prediction and the target
    rmspe = compute_rmspe(y, predict)

    print("the RMSPE of the baseline model (mean) is {}".format(rmspe.round(4)))

    
