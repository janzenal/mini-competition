import pandas as pd
import numpy as np
from math import sqrt

from data_cleaning import loading_data
from baseline_model import baseline
from linear_model import linear_regression
from extra_trees_model import extra_trees_regressor

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

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
                              "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval", "Date"])
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
def split_train_test(data):
    x_train = data.copy(deep=True).drop(columns=["Sales"])
    y_train = data.loc[:, "Sales"]
    
    # train, test split
    X_train,X_test,y_train,y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
    
    return X_train, X_test, y_train, y_test

# defining evaluation metric
def compute_rmspe(actual, prediction):
    rmspe = np.sqrt(np.mean(np.square(((actual - prediction) / actual)), axis=0)) 
    return rmspe

# our decision trees regressor model using the functions defined above
def decision_trees_regressor(data):
    #encode and transform
    encoded_data = encode(data)
    
    # deleting rows with null values and getting rid of some columns
    cleaned_data = delete_nulls(encoded_data)
    
    # splitting the data into features and target
    X_train, X_test, y_train, y_test = split_train_test(cleaned_data)
    
    # decision tree regression model
    dtc = tree.DecisionTreeRegressor(max_depth=5, random_state=42, criterion='mse')
    dtc.fit(X_train, y_train)
    predict = dtc.predict(X_test)
#     dtc.score(x_test, y_test)
    
    # computing the RMSPE of the difference between the prediction and the target
    rmspe = compute_rmspe(y_test, predict)

    print("the RMSPE of the decision trees regressor model is {}".format(rmspe.round(4)))