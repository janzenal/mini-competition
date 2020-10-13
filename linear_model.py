#importing libraries
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# mean encoding function
def mean_encode(data, col, on):
    group = data.groupby(col).mean()
    data.loc[:, col+'-original'] = data.loc[:, col]
    mapper = {k: v for k, v in zip(group.index, group.loc[:, on].values)}

    data.loc[:, col] = data.loc[:, col].replace(mapper)
    data.loc[:, col].fillna(value=np.mean(data.loc[:, col]), inplace=True)
    return data.loc[:, col]

# label encoding function
def label_encode(data, column):
    # instantiate the Label Encoder
    le = LabelEncoder()
    
    # encode the column
    data.loc[:, column] = le.fit_transform(data.loc[:, column])
    return data.loc[:, column]

# encode and transform the relevant columns
def encode(data):
    # encode columns
    data.loc[:, 'StoreType'] = label_encode(data, 'StoreType')
    data.loc[:, 'Assortment'] = label_encode(data, 'Assortment')
    data.loc[:, 'Store'] = mean_encode(data.copy(), 'Store', 'Sales')
    
    # transform the StateHoliday column into the numerical categories '1' and '0'
    data.loc[:, "StateHoliday"] = data.loc[:, "StateHoliday"].apply(lambda x: 1 if ((x == "a") or (x == "b")) else 0)
    return data

#drop columns and rows with null values
def delete_nulls(data):
    data = data.drop(columns=["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                              "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval", 
                              "Date"])
    data = data[~(data.loc[:, "DayOfWeek"].isnull()) &
                ~(data.loc[:, "Sales"].isnull()) &
                ~(data.loc[:, "Customers"].isnull()) &
                ~(data.loc[:, "Open"].isnull()) &
                ~(data.loc[:, "Promo"].isnull()) &
                ~(data.loc[:, "SchoolHoliday"].isnull()) &
                ~(data.loc[:, "CompetitionDistance"].isnull()) &
                ~(data.loc[:, "Sales"] == 0.0)]
    return data

# splitting features and target apart
def split(data):
    x_train = data.copy(deep=True).drop(columns=["Sales"])
    y_train = data.loc[:, "Sales"]
    return x_train, y_train

# defining evaluation metric
def compute_rmspe(actual, prediction):
    rmspe = np.sqrt(np.mean(np.square(((actual - prediction) / actual)), axis=0)) 
    return rmspe

# our linear regression model using the functions defined above
def linear_regression(data):
    #encode and transform
    encoded_data = encode(data)
    
    # deleting rows with null values and getting rid of some columns
    cleaned_data = delete_nulls(encoded_data)
    
    # splitting the data into features and target
    X, y = split(cleaned_data)
    
    # linear regression model
    lr = LinearRegression()
    lr.fit(X, y)
    predict = lr.predict(X)
    
    # computing the RMSPE of the difference between the prediction and the target
    rmspe = compute_rmspe(y, predict)

    print("the RMSPE of the linear regression model is {}".format(rmspe.round(4)))
