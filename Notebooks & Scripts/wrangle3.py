#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import datetime
import re
import requests
from bs4 import BeautifulSoup
import warnings

def prepare_data():
    # North Carolina:
    # https://www.newsobserver.com/news/coronavirus/article244546087.html
    # San Francisco:
    # https://data.sfgov.org/COVID-19/COVID-19-Cases-and-Deaths-Summarized-by-ZIP-Code-T/tef6-3vsw/data
    # Montgomery County (MD):
    # https://opendata.maryland.gov/Health-and-Human-Services/MD-COVID-19-Cases-by-ZIP-Code/ntd2-dqpx 
    # and
    # https://www.montgomerycountymd.gov/covid19/data/case-counts.html#zip-code
    nc_df = pd.read_csv('Data/Covid Data/nc_covid_data.csv')
    nc_df.columns = ['Zipcode', 'Case Counts', 'Death Counts']
    display(nc_df)
    sf_df = pd.read_csv('Data/Covid Data/sf_covid_data.csv')
    sf_df.columns = ['Zipcode', 'Case Counts', 'Death Counts']
    display(sf_df)
    mc_df = pd.read_csv('Data/Covid Data/montgomery_county_covid_data.csv')
    mc_df.columns = ['Zipcode', 'Case Counts', 'Death Counts']
    display(mc_df)
    frames = [nc_df, sf_df, mc_df]
    df = pd.concat(frames)
    return df

def scrape(z):
        z = str(z)
        source = requests.get('https://censusreporter.org/profiles/86000US{zip}-{zip}/'.format(zip=z)).text
        soup = BeautifulSoup(source, 'lxml')
        s = soup.findAll('script',type="text/javascript")[1]
        return str(s)

def parse(param, sp):
    param = '"'+param+'", '
    try:
        var = float(re.findall(param+'"values": {"this":\s*([+-]?[0-9]+\.[0-9]+)',sp)[0])
    except:
        var = 0

    return var

def getSocioDem(zipcodes):
    df = pd.DataFrame(columns = [ 'Zipcode', 'Population', 'Median age', 'Under 18(%)', '18 to 64(%)', '65 and over(%)', 'Male(%)', 'Female(%)', 'White(%)', "Black(%)", "Native(%)", "Asian(%)", "Islander(%)", "Two plus(%)", "Hispanic(%)", "Per capita income (USD)", "Median household income (USD)", "Below poverty line(%)",
                                'Mean travel time to work (Minutes)', 'Drove Alone (%)', 'Carpooled (%)', 'Public Transit (%)', 'Bicycle (%)', 'Walked (%)', 'Other (%)', 'Worked at home (%)', 'Number of households', 'Persons per household', 'Married (%)', 'Single (%)',
                                'Number of housing units', 'Occupied housing (%)', 'Vacant housing (%)', 'Owner Occupied (%)', 'Renter Occupied (%)', 'Median housing value',
                                'Moved Since Prev Year(%)', 'Same House Prev Year(%)', 'No Degree(%)', 'High School(%)', 'Some College(%)', "Bachelor's(%)", "Post-grad(%)", 'Foriegn Born Population(%)', 'Europe(%)', 'Asia(%)', 'Africa(%)', 'Oceania(%)', 'Latin America(%)', 'North America(%)'],
                                index = list(range(0,len(zipcodes))))

    i = 0
    for zip in zipcodes:
        try:
            s = scrape(zip)
        except:
            continue
        var = '"full_geoid": "86000US{z}", "total_population":'.format(z=str(zip))
        population = int(re.findall(var+'\s*([+-]?[0-9]+)',s)[0])

        median_age = parse("Median age",s)
        percent_under18 = parse("Under 18",s)
        percent_18to64 = parse("18 to 64",s)
        percent_65andOver = parse("65 and over",s)

        percent_male = parse("Male",s)
        percent_female = parse("Female",s)

        percent_white = parse("White",s)
        percent_black = parse("Black",s)
        percent_native = parse("Native",s)
        percent_asian = parse("Asian",s)
        percent_islander = parse("Islander",s)
        pecent_two_plus = parse("Two\+",s)
        percent_hispanic = parse("Hispanic",s)

        per_capita = parse("Per capita income", s)
        median_household_income = parse("Median household income",s)

        percent_below_poverty = parse("Persons below poverty line",s)

        mean_travel_time = parse("Mean travel time to work",s)
        drove_alone = parse("Drove alone",s)
        carpooled = parse("Carpooled",s)
        public_transit = parse("Public transit",s)
        bicycle = parse("Bicycle",s)
        walked = parse("Walked",s)
        other = parse("Other",s)
        worked_at_home = parse("Worked at home",s)

        number_of_households = parse("Number of households",s)
        persons_per_household = parse("Persons per household",s)

        married = parse("Married",s)
        single = parse("Single",s)

        number_of_housing_units = parse("Number of housing units",s)
        occupied_housing_units = parse("Occupied",s)
        vacant_housing_units = parse("Vacant",s)
        owner_housing_units = parse("Owner occupied",s)
        renter_housing_units = parse("Renter occupied",s)

        median_value_owner_occupied = parse("Median value of owner-occupied housing units",s)

        moved_since_previous_year = parse("Moved since previous year",s)
        same_house_year_ago = parse("Same house year ago",s)

        no_degree = parse("No degree",s)
        high_school = parse("High school",s)
        some_college = parse("Some college",s)
        bachelors = parse("Bachelor's",s)
        post_grad = parse("Post-grad",s)

        foriegn_born_pop = parse("Foreign-born population",s)
        europe = parse('Europe',s)
        asia = parse('Asia',s)
        africa = parse('Africa',s)
        oceania = parse('Oceania',s)
        latin_america = parse('Latin America',s)
        north_america = parse('North America',s)

        df.iloc[i] = [zip, population, median_age, percent_under18, percent_18to64, percent_65andOver, percent_male, percent_female, percent_white, percent_black, percent_native, percent_asian, percent_islander, pecent_two_plus, percent_hispanic, per_capita, median_household_income, percent_below_poverty, mean_travel_time,
        drove_alone, carpooled, public_transit, bicycle, walked, other, worked_at_home, number_of_households, persons_per_household, married, single, number_of_housing_units, occupied_housing_units, vacant_housing_units, owner_housing_units, renter_housing_units,
        median_value_owner_occupied, moved_since_previous_year, same_house_year_ago, no_degree, high_school, some_college, bachelors, post_grad, foriegn_born_pop, europe, asia, africa, oceania, latin_america, north_america]

        i+=1
    return df

def get_data():
    # scrape social demographic data 
    df = prepare_data()
    zipcodes = df.Zipcode.tolist()
    socio_df = getSocioDem(zipcodes)
    merge_df = pd.merge(socio_df, df, how='inner', on='Zipcode')
    # drop data rows that have 0 population
    zero_rows = merge_df[merge_df['Population'] == 0].index.tolist()
    merge_df = merge_df.drop(zero_rows)
    # drop data rows that have 0 mean household income
    merge_df = merge_df.drop(merge_df[merge_df['Median household income (USD)'] == 0].index.tolist())
    merge_df['Death Counts(Per 1000)'] = (merge_df['Death Counts'] / merge_df['Population']) * 1000
    merge_df['Case Counts(Per 1000)'] = (merge_df['Case Counts'] / merge_df['Population']) * 1000
    return merge_df

