import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
# %matplotlib inline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error , r2_score , explained_variance_score, accuracy_score
from sklearn.model_selection import cross_val_score
import xgboost
from xgBoost_Tuning import *
import warnings
warnings.filterwarnings('ignore')


def xgBoost_analysis():
    data = pd.read_csv("AllData-Covid-SocioDemographics-Cases-Deaths.csv")
    cormat = data.corr().abs()
    cormat = cormat[cormat>0.1]
    feature_importance = round(cormat,2).dropna(subset=['Death Counts(Per 1000)'])['Death Counts(Per 1000)'].to_frame().reset_index()
    feature_importance = feature_importance[:-2]
    important_features = list(feature_importance['index'])
    data_input = data.iloc[:,:-2]
    data_input = data_input.set_index('Zipcode')
    data_input = data_input[important_features]

    X_train, X_test, y_train, y_test  = trim_and_split(data_input)
    X_train = X_train.drop('Death Counts', 1)
    X_test = X_test.drop('Death Counts', 1)
    tuned_model = XGBRegressor(colsample_bytree=0.9, gamma=0.0, max_depth=5, n_estimators=80,
             objective='reg:squarederror', subsample=0.9)
    tuned_model.fit(X_train, y_train)
    y_pred_updated = tuned_model.predict(X_test)
    absolute_error = abs(y_pred_updated - y_test)
    abs_error = round(np.mean(absolute_error), 2)
    MSE = mean_squared_error(y_test, y_pred_updated)
    R2_score = r2_score(y_test, y_pred_updated)

    train_error = abs((tuned_model.predict(X_train)) - (y_train))
    test_error = abs((tuned_model.predict(X_test)) - (y_test))

    fig,ax = plt.subplots(figsize=(6,6))
    ax.scatter(x = (tuned_model.predict(X_test)), y = (y_test), s = 70)

    lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]

    plt.rcParams.update({'font.size': 15})
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Covid-19 Related Deaths per 1000\n(predicted by tuned XGBoost model)")
    print(" ")
    print("Mean absolute training error (deaths per thousand): ", np.mean(train_error))
    print("Mean absolute test error (deaths per thousand): ", np.mean(test_error))
    
    



