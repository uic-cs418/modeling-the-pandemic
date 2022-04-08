# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:14:51 2022

@author: fanta
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

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