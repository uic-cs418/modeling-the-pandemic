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
    df = pd.read_csv("AllData-Covid-SocioDemographics-Cases-Deaths.csv")
    y = df['Death Counts(Per 1000)']
    X = df.drop('Death Counts(Per 1000)', 1)
    X = X.drop('Death Counts', 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    tuned_model = XGBRegressor(colsample_bytree=0.9, gamma=0.0, max_depth=5, n_estimators=80,
             objective='reg:squarederror', subsample=0.9)
    tuned_model.fit(X_train, y_train)

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
    
    



