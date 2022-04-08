import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas import GeoDataFrame
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.filterwarnings('ignore')

class Vis:

    def lineChart(self, zipcodes, covid, caseCountByDate):
        deathCountByDate = covid.groupby('Date of Death').size().to_frame().reset_index().rename(columns={0: 'Death Counts'})

        caseCountByDate = caseCountByDate.fillna(0)
        caseCountByDate['Week Start'] = pd.to_datetime(caseCountByDate['Week Start']).dt.date
        caseCountByDate['Week End'] = pd.to_datetime(caseCountByDate['Week End']).dt.date
        caseCountByDate = caseCountByDate[caseCountByDate['ZIP Code'] != 'Unknown']
        caseCountByDate['ZIP Code'] = caseCountByDate['ZIP Code'].astype(str).astype(int)
        caseCountByDate = caseCountByDate[caseCountByDate['ZIP Code'].isin(zipcodes)]
        caseCountByDate = caseCountByDate.rename(columns={'ZIP Code': 'Zipcode'})
        caseCountByDate = caseCountByDate.sort_values(by='Week End')
        caseCountByDate = caseCountByDate[['Week End','Cases - Weekly', 'Deaths - Weekly']]
        caseCountByDate = caseCountByDate.rename(columns={'Week End': 'Date', 'Cases - Weekly': 'Cases'})
        caseCountByDate = caseCountByDate.groupby('Date').sum().reset_index()

        fig, ((ax1, ax2)) = plt.subplots(nrows=2, ncols=1, figsize=(30,10))

        sns.lineplot(ax=ax1, data=deathCountByDate, x="Date of Death", y="Death Counts", color='r')
        ax1.set_xlabel("Date")
        ax1.set_title("Covid Daily Death Counts in Chicago")

        sns.lineplot(ax=ax2, data=caseCountByDate, x="Date", y="Cases", color='Purple')
        current_values = plt.gca().get_yticks()
        plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
        ax2.set_title("Covid Daily Case Counts in Chicago")

        fig.tight_layout()
    
    def scatterPlot(self, mergedSocCovid):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(25,10))
        fig.suptitle('Correlation of Sociodemographic factors with Covid Death Cases',fontsize=15)

        sns.regplot(ax=ax1, data=mergedSocCovid, x="Median household income (USD)", y="Death Counts(Per 1000)")
        sns.regplot(ax=ax2, data=mergedSocCovid, x="Per capita income (USD)", y="Death Counts(Per 1000)")
        sns.regplot(ax=ax3, data=mergedSocCovid, x="Below poverty line(%)", y="Death Counts(Per 1000)")
        sns.regplot(ax=ax4, data=mergedSocCovid, x="Median housing value", y="Death Counts(Per 1000)")

    def featureImportance(self, mergedSocCovid):
        cormat = mergedSocCovid.corr().abs()
        cormat = cormat[cormat>0.5]
        feature_importance = round(cormat,2).dropna(subset=['Death Counts(Per 1000)'])['Death Counts(Per 1000)'].to_frame().reset_index()
        feature_importance = feature_importance[:-2]
        feature_importance = feature_importance.sort_values(by='Death Counts(Per 1000)')
        plt.figure(figsize=(15,8))
        ax = sns.barplot(data=feature_importance, y='index', x='Death Counts(Per 1000)', palette='flare')
        plt.xlabel('Feature Importance (Scale: 0-1)')
        plt.ylabel('')
        plt.title('Feature Importance of Sociodemographic Factors w.r.t Covid Death Cases',fontsize=15)


    
    def geographicPlot(self, gdf, mergedSocCovid):
        gdf = gdf.rename(columns={"zip": "Zipcode"})
        gdf['Zipcode'] = gdf['Zipcode'].astype(str).astype(int)
        df = pd.merge(mergedSocCovid, gdf, how='inner', on = 'Zipcode')
        gdf = gpd.GeoDataFrame(df)
        fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(40,40))
        ax1.set_title('Covid Case Rates in Chicago Neighborhoods', fontsize=15)
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        plt.rcParams.update({'font.size': 8})
        gdf.apply(lambda x: ax1.annotate(text=x['Zipcode'], color='black', xy=x.geometry.centroid.coords[0], ha='right'), axis=1)
        gdf.plot(column = 'Case Counts(Per 1000)', ax=ax1, legend=True, cax=cax1, legend_kwds={'label': 'Case Counts(Per 1000)'});

        ax2.set_title('Covid Death Rates in Chicago Neighborhoods', fontsize=15)
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.1)
        plt.rcParams.update({'font.size': 8})
        gdf.apply(lambda x: ax2.annotate(text=x['Zipcode'], color='black', xy=x.geometry.centroid.coords[0], ha='right'), axis=1)
        gdf.plot(column = 'Death Counts(Per 1000)', ax=ax2, legend=True, cax=cax2, legend_kwds={'label': 'Death Counts(Per 1000)'});