# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:52:34 2022

@author: fanta
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 5000, num = 50)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 100, num = 50)]
max_depth.append(None)
min_samples_split = [int(x) for x in np.linspace(2, 11, num = 10)]
min_samples_leaf = [int(x) for x in np.linspace(1, 10, num = 10)]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

def optimize_rf(train_features, train_labels, random_grid):
    rf = RandomForestRegressor(random_state = 17)
    rf_random = RandomizedSearchCV(estimator = rf,
                               param_distributions = random_grid,
                               n_iter = 100, 
                               cv = RepeatedKFold(n_splits = 2, n_repeats = 5),
                               verbose = 100, random_state = 17, n_jobs = -1)
    rf_random.fit(train_features, train_labels)
                          
    return rf_random
    
    