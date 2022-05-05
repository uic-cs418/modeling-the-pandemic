# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:14:51 2022

@author: fanta
"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from rf_analysis import *
import pandas as pd

def scale_df(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled

def PCA_2(X, random_state = 17):
    pca_2 = PCA(n_components = 2, random_state = random_state)
    X_scaled = scale_df(X)
    pca_2.fit(X_scaled)
    X_pca_2 = pca_2.transform(X_scaled)
    return pca_2, X_pca_2

def plot_pca(pca_data, pca_model, labels, title, cmap = "icefire"):
    pc1 = pca_data[:,0]
    pc2 = pca_data[:,1]
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(10,7))
    plt.scatter(x = pc1, y = pc2, s = 70,
                c = labels, cmap = cmap)
    sm = plt.cm.ScalarMappable(cmap = cmap,
                               norm = plt.Normalize(vmin = np.min(labels), 
                                                    vmax = np.max(labels)))
    sm._A = []
    cbar = plt.colorbar(sm)
    
    perc_explained = pca_model.explained_variance_ratio_*100
    expl_1 = round(perc_explained[0],2)
    expl_2 = round(perc_explained[1],2)
    
    plt.xlabel("PC 1 (" + str(expl_1) + "% of Variance)")
    plt.ylabel("PC 2 (" + str(expl_2) + "% of Variance)")
    plt.title(title)
    plt.show()
    
def plot_pca_cat_labeled(pca_data, pca_model, labels, title, data, cols):
    pc1 = pca_data[:,0]
    pc2 = pca_data[:,1]
    #print(pc1, pc2)
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(10,7))
    plt.scatter(x = pc1, y = pc2, s = 70,
                c = cols, alpha = .65)

    
    perc_explained = pca_model.explained_variance_ratio_*100
    expl_1 = round(perc_explained[0],2)
    expl_2 = round(perc_explained[1],2)
    
    plt.xlabel("PC 1 (" + str(expl_1) + "% of Variance)")
    plt.ylabel("PC 2 (" + str(expl_2) + "% of Variance)")
    plt.title(title)
    for i in range(0, len(data.loc[:,"Zipcode"])):
        print(pc1[i])
        print(pc2[i])
        print(str(data.loc[:,"Zipcode"][i]))
        plt.text(pc1[i], pc2[i], str(data.loc[:,"Zipcode"][i]))
    plt.show()
    
def pltcolor(lst):
    cols=[]
    for l in lst:
        if l=='NY':
            cols.append(1)
        elif l=='TX':
            cols.append(2)
        elif l=='WI':
            cols.append(3)
        elif l =='IL':
            cols.append(4)
        else:
            cols.append(5)
    return np.array(cols)
# Create the colors list using the function above
#cols=pltcolor(X_train.loc[:,"state"])

def plot_pca_cat(pca_data, pca_model, labels, title, cols):
    pc1 = pca_data[:,0]
    pc2 = pca_data[:,1]
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(10,7))
    plt.scatter(x = pc1, y = pc2, s = 70,
                c = cols, alpha = .65)

    
    perc_explained = pca_model.explained_variance_ratio_*100
    expl_1 = round(perc_explained[0],2)
    expl_2 = round(perc_explained[1],2)
    
    plt.xlabel("PC 1 (" + str(expl_1) + "% of Variance)")
    plt.ylabel("PC 2 (" + str(expl_2) + "% of Variance)")
    plt.title(title)
    

    plt.show()
    
def form_PCA(df):
    Numeric_socdem = df.loc[:,"Median age":"North America(%)"]

    pca2, xpca2 = PCA_2(Numeric_socdem)

    return pca2, xpca2

def pca_analysis():
    data = pd.read_csv("AllData-Covid-SocioDemographics-Cases-Deaths.csv")

    X_train, X_test, y_train, y_test = trim_and_split(data)
    
    pca2, xpca2 = form_PCA(X_train)
    
    plot_pca(xpca2, pca2, y_train, "Principal Components plot\nColored by COVID-19 related deaths per 1000")