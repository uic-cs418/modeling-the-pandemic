import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error , r2_score , explained_variance_score, accuracy_score
from sklearn.model_selection import cross_val_score
import xgboost


    
def trim_and_split(df, remove_0s = True, remove_outliers = True):
    if remove_0s:
        df = df[df.loc[:,"Death Counts(Per 1000)"] != 0]
    else:
        pass
    features = df.iloc[:,:-1]
    labels = df.iloc[:,-3]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)
    if remove_outliers:
        
        top5 = np.percentile(y_train, 95)
    
        X_train_95 = X_train[y_train <= top5]
        y_train_95 = y_train[y_train <= top5]

        X_test_95 = X_test[y_test <= top5]
        y_test_95 = y_test[y_test <= top5]
        
        return X_train_95, X_test_95, y_train_95, y_test_95
    else:
        return X_train, X_test, y_train, y_test

def test_train_split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return(X_train, X_test, y_train, y_test)

def default_xGBoost():
    model = XGBRegressor(features="sqrt")
    return model
    