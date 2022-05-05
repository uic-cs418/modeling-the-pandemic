# -*- coding: utf-8 -*-
"""
Created on Thu May  5 00:52:11 2022

@author: fanta
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from random_forest import *
from optimize_rfr_hyperparams2 import *

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

def select_relevant(X):
    return(X.loc[:,"Median age":"North America(%)"])


def rf_analysis():
    data = pd.read_csv("socio-demographic-and-death-counts(combined).csv")
    X_train, X_test, y_train, y_test = trim_and_split(data, True, True)
    X_train2, X_test2 = select_relevant(X_train), select_relevant(X_test)
    rf_model = rfr_default(X_train2, y_train)
    train_error = abs((rf_model.predict(X_train2)) - (y_train))
    test_error = abs((rf_model.predict(X_test2)) - (y_test))
    
    
    return {"model": rf_model,
            "train_error": train_error,
            "test_error": test_error}

def plot_obs_exp(model, x, y):
    fig,ax = plt.subplots()

    ax.scatter(x = (model.predict(x)), y = (y), s = 70)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Covid-19 Related Deaths per 1000")
    ax.show()