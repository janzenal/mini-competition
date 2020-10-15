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
from sklearn import tree
import xgboost as xgb

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
    #data.loc[:, 'DayOfWeek'] = mean_encode(data.copy(), 'DayOfWeek', 'Sales')
    #data.loc[:, 'WeekOfMonth'] = mean_encode(data.copy(), 'WeekOfMonth', 'Sales')
    #data.loc[:, 'Month'] = mean_encode(data.copy(), 'Month', 'Sales')
    
    # transform the StateHoliday column into the numerical categories '1' and '0'
    data.loc[:, "StateHoliday"] = data.loc[:, "StateHoliday"].apply(lambda x: 0 if ((x == "a") or (x == "b") or (x == "c")) else 1)
    return data

#drop columns and rows with null values
def delete_nulls(data):
    data = data.drop(columns=["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                              "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval", "Date", "Customers"])
    data = data[~(data.loc[:, "DayOfWeek"].isnull()) &
                ~(data.loc[:, "Open"].isnull()) &
                ~(data.loc[:, "Promo"].isnull()) &
                ~(data.loc[:, "Promo2"].isnull()) &
                ~(data.loc[:, "SchoolHoliday"].isnull()) &
                ~(data.loc[:, "StateHoliday"].isnull()) &
                ~(data.loc[:, "CompetitionDistance"].isnull())]
    if 'Sales' in data.columns:
        data = data[~(data.loc[:, "Sales"].isnull()) &
                    ~(data.loc[:, "Sales"] == 0.0)]
    #data.loc[:, "DayOfWeek"] = data.loc[:, "DayOfWeek"]**3
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

# splitting features and target back apart
def split(data):
    x_train = data.copy(deep=True).drop(columns=["Sales"])
    y_train = data.loc[:, "Sales"]
    return x_train, y_train

# defining evaluation metric
def compute_rmspe(actual, prediction):
    rmspe = np.sqrt(np.mean(np.square(((actual - prediction) / actual)), axis=0)) 
    return rmspe

# using the mean of the entire training set as a first prediction
def lazy_estimator(target):
    lazy_estimator_predictions = pd.DataFrame(target.copy())
    lazy_estimator_predictions.loc[:, 'lazy_predicted_price'] = target.mean()
    predict = lazy_estimator_predictions.loc[:, 'lazy_predicted_price']
    return predict

# our baseline model using the functions defined above
def baseline(data1, data2):
    # deleting rows with null values and getting rid of some columns
    cleaned_data1 = delete_nulls(data1)
    cleaned_data2 = delete_nulls(data2)
    
    # splitting the data into features and target
    X1, y1 = split(cleaned_data1)
    X2, y2 = split(cleaned_data2)
    
    # calculating the prediction which is simply the mean here
    predict = lazy_estimator(y1)
    
    # computing the RMSPE of the difference between the prediction and the target
    rmspe = compute_rmspe(y2, predict)*100

    print("the RMSPE of the baseline model (mean) is {}%".format(rmspe.round(4)))

# our linear regression model using the functions defined above
def linear_regression(data1, data2):
    # encode and transform
    encoded_data1 = encode(data1)
    encoded_data2 = encode(data2)
    
    # deleting rows with null values and getting rid of some columns
    cleaned_data1 = delete_nulls(encoded_data1)
    cleaned_data2 = delete_nulls(encoded_data2)
        
    # splitting the data into features and target
    #X_train, X_test, X_val, y_train, y_test, y_val = split_train_test(cleaned_data)
    # splitting the data into features and target
    X1, y1 = split(cleaned_data1)
    X2, y2 = split(cleaned_data2)
    
    # scaling the features for train and test set
    X1 = scale(X1)
    X2 = scale(X2)
    
    # linear regression model
    lr = LinearRegression()
    lr.fit(X1, y1)
    predict = lr.predict(X2)
    
    # computing the RMSPE of the difference between the prediction and the target
    rmspe = compute_rmspe(y2, predict)*100

    print("the RMSPE of the linear regression model is {}%".format(rmspe.round(4)))

# our extra trees model using the functions defined above
def extra_trees_regressor(data1, data2):
    # encode and transform
    encoded_data1 = encode(data1)
    encoded_data2 = encode(data2)
    
    # deleting rows with null values and getting rid of some columns
    cleaned_data1 = delete_nulls(encoded_data1)
    cleaned_data2 = delete_nulls(encoded_data2)
        
    # splitting the data into features and target
    #X_train, X_test, X_val, y_train, y_test, y_val = split_train_test(cleaned_data)
    # splitting the data into features and target
    X1, y1 = split(cleaned_data1)
    X2, y2 = split(cleaned_data2)
    
    # linear regression model
    et = ExtraTreesRegressor(criterion='mse',n_estimators=20, max_depth=9, n_jobs=-1)
    et.fit(X1, y1)
    predict = et.predict(X2)
    
    # computing the RMSPE of the difference between the prediction and the target
    rmspe = compute_rmspe(y2, predict)*100

    print("the RMSPE of the extra trees model is {}%".format(rmspe.round(4)))

# our random forest model using the functions defined above
def random_forest_regressor(data1, data2):
    # encode and transform
    encoded_data1 = encode(data1)
    encoded_data2 = encode(data2)
    
    # deleting rows with null values and getting rid of some columns
    cleaned_data1 = delete_nulls(encoded_data1)
    cleaned_data2 = delete_nulls(encoded_data2)
        
    # splitting the data into features and target
    #X_train, X_test, X_val, y_train, y_test, y_val = split_train_test(cleaned_data)
    # splitting the data into features and target
    X1, y1 = split(cleaned_data1)
    X2, y2 = split(cleaned_data2)
    
    # linear regression model
    rt = RandomForestRegressor(criterion='mse',n_estimators=80,max_depth=20,n_jobs=-1)
    rt.fit(X1, y1)
    predict = rt.predict(X2)
    
    # computing the RMSPE of the difference between the prediction and the target
    rmspe = compute_rmspe(y2, predict)*100

    print("the RMSPE of the random forest model is {}%".format(rmspe.round(4)))

# our decision trees regressor model using the functions defined above
def decision_trees_regressor(data):
    # encode and transform
    encoded_data = encode(data)
    
    # deleting rows with null values and getting rid of some columns
    cleaned_data = delete_nulls(encoded_data)
    
    # splitting the data into features and target
    X_train, X_test, X_val, y_train, y_test, y_val = split_train_test(cleaned_data)
    
    # scaling the features for train and test set
    #X_train = scale(X_train)
    #X_test = scale(X_test)
    
    # decision tree regression model
    dtc = tree.DecisionTreeRegressor(max_depth=5, random_state=42, criterion='mse')
    dtc.fit(X_train, y_train)
    predict = dtc.predict(X_test)
    
    # computing the RMSPE of the difference between the prediction and the target
    rmspe = compute_rmspe(y_test, predict)*100

    print("the RMSPE of the decision trees regressor model is {}%".format(rmspe.round(4)))
    
# our boosted trees regressor model using the functions defined above
def xgb_regressor(data1, data2):
    # encode and transform
    encoded_data1 = encode(data1)
    encoded_data2 = encode(data2)
    
    # deleting rows with null values and getting rid of some columns
    cleaned_data1 = delete_nulls(encoded_data1)
    cleaned_data2 = delete_nulls(encoded_data2)
        
    # splitting the data into features and target
    #X_train, X_test, X_val, y_train, y_test, y_val = split_train_test(cleaned_data)
    # splitting the data into features and target
    X1, y1 = split(cleaned_data1)
    X2, y2 = split(cleaned_data2)    
    
    # decision tree regression model
    xgbr = xgb.XGBRegressor(max_depth=6,learning_rate=0.3,n_estimators=500,n_jobs=-1)
    xgbr.fit(X1, y1)
    predict = xgbr.predict(X2)

    
    # computing the RMSPE of the difference between the prediction and the target
    rmspe = compute_rmspe(y2, predict)*100

    print("the RMSPE of the boosted trees model is {}%".format(rmspe.round(4)))