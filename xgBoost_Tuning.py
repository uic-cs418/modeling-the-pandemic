import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error , r2_score , explained_variance_score, accuracy_score
from sklearn.model_selection import cross_val_score
import xgboost
from xgBoost import *

param_test = {
    'n_estimators':range(20,100,20),
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2),
    'gamma':[i/10.0 for i in range(0,5)],
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
    }

def tune_xgBoost(X_train, y_train):
    gsearch = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1,min_samples_split=500,min_samples_leaf=50,random_state=10, objective="reg:squarederror"),  param_grid = param_test,n_jobs=10, cv=5)
    gsearch.fit(X_train, y_train)
    tuned_params = gsearch.best_params_
    tuned_params["objective"] = "reg:squarederror"
    return XGBRegressor(**tuned_params)
    