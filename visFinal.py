import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas import GeoDataFrame
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 15})

def covidDeath():
    df = pd.read_csv("Data/Covid Data/Medical_Examiner_Case_Archive_-_COVID-19_Related_Deaths.csv")
    zip = pd.read_csv("Data/Zip_Codes.csv")
    zipcodes = zip.ZIP.to_list()
    zipcodes.remove(60666) #OHare Airport!!

    
    df = df[df['Residence Zip'].notna()]
    sep = '-'
    df['Residence Zip'] = df['Residence Zip'].apply(lambda x: x.split(sep, 1)[0])
    df['Residence Zip'] = df['Residence Zip'].astype(str).astype(int)
    df = df[df['Residence Zip'].isin(zipcodes)]
    df['Date of Death'] = pd.to_datetime(df['Date of Death'])
    df['Date of Death'] = df['Date of Death'].dt.date
    df = df[df['Date of Death'] <= datetime.date(2022, 3, 5)]
    df = df.sort_values(by=['Date of Death'])
    df = df[(df['Manner of Death']!='ACCIDENT') & (df['Manner of Death']!='SUICIDE')]
    df = df[df['COVID Related'] == True]
    df = df.rename(columns={"Residence Zip": "Zipcode"})

    df = df.drop(['Case Number','Date of Incident','Manner of Death','Residence City', 'Primary Cause', 'Primary Cause Line A', 'Primary Cause Line B',
        'Primary Cause Line C', 'Secondary Cause', 'Gun Related', 'COVID Related', 'Opioid Related', 'Commissioner District',
        'Incident Address', 'Incident City', 'Incident Zip Code', 'longitude',
        'latitude', 'location', 'OBJECTID', 'Chicago Ward',
        'Chicago Community Area'],axis=1)

    return df

def lineChart():
    plt.rcParams.update({'font.size': 15})
    zip = pd.read_csv("Data/Zip_Codes.csv")
    zipcodes = zip.ZIP.to_list()
    zipcodes.remove(60666) #OHare Airport!!

    dfDeath = covidDeath()
    deathCountByDate = dfDeath.groupby('Date of Death').size().to_frame().reset_index().rename(columns={0: 'Death Counts'})

    caseCountByDate = pd.read_csv('Data/Covid Data/COVID-19_Cases__Tests__and_Deaths_by_ZIP_Code.csv')
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

def scatterPlot(mergedSocCovid, city):
    plt.rcParams.update({'font.size': 13})
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(25,10))
    fig.suptitle('Correlation of Sociodemographic factors with Covid Death Cases: '+ city,fontsize=15)

    sns.regplot(ax=ax1, data=mergedSocCovid, x="Median household income (USD)", y="Death Counts(Per 1000)")
    sns.regplot(ax=ax2, data=mergedSocCovid, x="Per capita income (USD)", y="Death Counts(Per 1000)")

    print()
    sns.regplot(ax=ax3, data=mergedSocCovid, x="Below poverty line(%)", y="Death Counts(Per 1000)")
    sns.regplot(ax=ax4, data=mergedSocCovid, x="Median housing value", y="Death Counts(Per 1000)")

def scatterPlotComp(All, Ny, Chi):
    plt.rcParams.update({'font.size': 13})
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(35, 5))

    ax1.title.set_text('\nAll Data')
    sns.regplot(ax=ax1, data=All, x="Median household income (USD)", y="Death Counts(Per 1000)")
    ax2.title.set_text('\nNew York')
    sns.regplot(ax=ax2, data=Ny, x="Median household income (USD)", y="Death Counts(Per 1000)")
    ax3.title.set_text('\nChicago')
    sns.regplot(ax=ax3, data=Chi, x="Median household income (USD)", y="Death Counts(Per 1000)")


def featureImportance(mergedSocCovid,cor):
    plt.rcParams.update({'font.size': 13})
    cormat = mergedSocCovid.corr().abs()
    cormat = cormat[cormat>cor]
    feature_importance = round(cormat,2).dropna(subset=['Death Counts(Per 1000)'])['Death Counts(Per 1000)'].to_frame().reset_index()
    feature_importance = feature_importance[:-4]
    feature_importance = feature_importance.sort_values(by='Death Counts(Per 1000)')
    plt.figure(figsize=(15,8))
    ax = sns.barplot(data=feature_importance, y='index', x='Death Counts(Per 1000)', palette='flare')
    plt.xlabel('Feature Importance (Scale: 0-1)')
    plt.ylabel('')
    plt.title('Feature Importance of Sociodemographic Factors w.r.t Covid Death Cases',fontsize=15)



def geographicPlot(mergedSocCovid):
    gdf = gpd.read_file("Data/Boundaries - ZIP Codes.geojson")
    gdf = gdf.rename(columns={"zip": "Zipcode"})
    gdf['Zipcode'] = gdf['Zipcode'].astype(str).astype(int)
    df = pd.merge(mergedSocCovid, gdf, how='inner', on = 'Zipcode')
    gdf = gpd.GeoDataFrame(df)
    plt.rcParams.update({'font.size': 14})
    fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(40,40))
    ax1.set_title('Covid Case Rates in Chicago Neighborhoods', fontsize=20)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.1)
    plt.rcParams.update({'font.size': 14})
    gdf.apply(lambda x: ax1.annotate(text=x['Zipcode'], fontsize=13, color='black', xy=x.geometry.centroid.coords[0], ha='right'), axis=1)
    gdf.plot(column = 'Case Counts(Per 1000)', ax=ax1, cmap='crest', legend=True, cax=cax1, legend_kwds={'label': 'Case Counts(Per 1000)'});

    ax2.set_title('Covid Death Rates in Chicago Neighborhoods', fontsize=20)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.1)
    plt.rcParams.update({'font.size': 14})
    gdf.apply(lambda x: ax2.annotate(text=x['Zipcode'], fontsize=13, color='black', xy=x.geometry.centroid.coords[0], ha='right'), axis=1)
    gdf.plot(column = 'Death Counts(Per 1000)', ax=ax2, cmap='crest', legend=True, cax=cax2, legend_kwds={'label': 'Death Counts(Per 1000)'});