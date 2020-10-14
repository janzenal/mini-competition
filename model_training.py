#importing libraries
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

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
    data.loc[:, 'StoreType'] = mean_encode(data.copy(), 'StoreType', 'Sales')
    data.loc[:, 'Assortment'] = mean_encode(data.copy(), 'Assortment', 'Sales')
    data.loc[:, 'Store'] = mean_encode(data.copy(), 'Store', 'Sales')
    data.loc[:, 'DayOfWeek'] = mean_encode(data.copy(), 'DayOfWeek', 'Sales')
    
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

# feature scaling
def scale(data):
    std_scale = preprocessing.StandardScaler().fit(data)
    df_std = std_scale.transform(data)
    return df_std

# splitting features and target apart
def split_train_test(data):
    x_train = data.copy(deep=True).drop(columns=["Sales"])
    y_train = data.loc[:, "Sales"]
    
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    
    # train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=1 - train_ratio, shuffle=False)
    
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=False) 
    
    return X_train, X_test, X_val, y_train, y_test, y_val

# defining evaluation metric
def compute_rmspe(actual, prediction):
    rmspe = np.sqrt(np.mean(np.square(((actual - prediction) / actual)), axis=0)) 
    return rmspe

# our linear regression model using the functions defined above
def linear_regression(data, est=1, maxdep=1):
    #encode and transform
    encoded_data = encode(data)
    
    # deleting rows with null values and getting rid of some columns
    cleaned_data = delete_nulls(encoded_data)
        
    # splitting the data into features and target
    X_train, X_test, X_val, y_train, y_test, y_val = split_train_test(cleaned_data)
    
    # scaling the features for train and test set
    X_train = scale(X_train)
    X_test = scale(X_test)
    
    # linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    predict = lr.predict(X_val)
    
    # computing the RMSPE of the difference between the prediction and the target
    rmspe = compute_rmspe(y_val, predict)

    return rmspe.round(4)

# our extra trees model using the functions defined above
def extra_trees_regressor(data, est, maxdep):
    #encode and transform
    encoded_data = encode(data)
    
    # deleting rows with null values and getting rid of some columns
    cleaned_data = delete_nulls(encoded_data)
    
    # splitting the data into features and target
    X_train, X_test, X_val, y_train, y_test, y_val = split_train_test(cleaned_data)
    
    # linear regression model
    et = ExtraTreesRegressor(criterion='mse',n_estimators=est, max_depth=maxdep, n_jobs=-1)
    et.fit(X_train, y_train)
    predict = et.predict(X_val)
    
    # computing the RMSPE of the difference between the prediction and the target
    rmspe = compute_rmspe(y_val, predict)

    return rmspe.round(4)

# our random forest model using the functions defined above
def random_forest_regressor(data, est, maxdep):
    #encode and transform
    encoded_data = encode(data)
    
    # deleting rows with null values and getting rid of some columns
    cleaned_data = delete_nulls(encoded_data)
    
    # splitting the data into features and target
    X_train, X_test, X_val, y_train, y_test, y_val = split_train_test(cleaned_data)
    
    # linear regression model
    rt = RandomForestRegressor(criterion='mse',n_estimators=est,max_depth=maxdep,n_jobs=-1)
    rt.fit(X_train, y_train)
    predict = rt.predict(X_val)
    
    # computing the RMSPE of the difference between the prediction and the target
    rmspe = compute_rmspe(y_val, predict)

    return rmspe.round(4)

