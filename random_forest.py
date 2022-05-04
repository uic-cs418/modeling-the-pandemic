# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 18:17:54 2022

@author: fanta
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def format_data(data, label_col):
    data_input = data.iloc[:,:-5]
    data_input = data_input.set_index('Zipcode')
    features = np.array(data_input)
    #feature_list = list(data_input.columns)
    labels = np.array(data[label_col])
    return features, labels

def filter_by_cor(data, label_col, min_coeff = 0.5):
    cormat = data.corr().abs()
    cormat = cormat[cormat>min_coeff]
    feature_importance = round(cormat,2).dropna(subset=['Death Counts(Per 1000)'])['Death Counts(Per 1000)'].to_frame().reset_index()
    feature_importance = feature_importance[:-2]
    important_features = list(feature_importance['index'])
    data_input = data.iloc[:,:-4]
    data_input = data_input.set_index('Zipcode')
    data_input = data_input[important_features]
    features = np.array(data_input)
    labels = np.array(data[label_col])
    return features, labels

def split(features, labels):
   train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = 0) 
   return train_features, test_features, train_labels, test_labels

def rfr_default(features, labels, random_state = 0):
    rf = RandomForestRegressor(n_estimators = 100, random_state = random_state)
    rf.fit(features, labels)
    return rf

def rfr_custom(features, labels, params, random_state = 0):
    rf = RandomForestRegressor(n_estimators = params['n_estimators'],
                               bootstrap = params['bootstrap'],
                               max_depth = params['max_depth'],
                               max_features = params['max_features'],
                               min_samples_leaf = params['min_samples_leaf'],
                               min_samples_split = params['min_samples_split'],
                               random_state = random_state)
    rf.fit(features, labels)
    return rf
