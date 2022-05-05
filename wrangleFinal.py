import numpy as np
import pandas as pd
import datetime
import re
import requests
from bs4 import BeautifulSoup
import os
import warnings
warnings.filterwarnings('ignore')

def wrangleCovidCaseData(df,zipcodes):
    df['Week Start'] = pd.to_datetime(df['Week Start']).dt.date
    df['Week End'] = pd.to_datetime(df['Week End']).dt.date
    df = df[df['Week End'] == datetime.date(2022, 3, 5)]
    df['Cases - Cumulative'] = df['Cases - Cumulative'].apply(np.int64)
    df = df[['ZIP Code', 'Cases - Cumulative']]
    df = df[df['ZIP Code'] != 'Unknown']
    df['ZIP Code'] = df['ZIP Code'].astype(str).astype(int)
    df = df[df['ZIP Code'].isin(zipcodes)]
    df = df.rename(columns={'ZIP Code': 'Zipcode'})
    df = df.rename(columns={'Cases - Cumulative': 'Case Counts'})

    return df

def wrangleCovidDeathData(df, zipcodes):
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

    df = df.groupby('Zipcode').size().to_frame().reset_index().rename(columns={0: 'Death Counts'})

    return df

def scrape(z):
    z = str(z)
    source = requests.get('https://censusreporter.org/profiles/86000US{zip}-{zip}/'.format(zip=z)).text
    soup = BeautifulSoup(source, 'lxml')
    s = soup.findAll('script',type="text/javascript")[1].getText()
    return s

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
        s = scrape(zip)

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

def mergeChi(soc, deaths, cases):
    mergedSocCovid = pd.merge(soc, deaths, how='inner', on = 'Zipcode')
    mergedSocCovid['Death Counts(Per 1000)'] = (mergedSocCovid['Death Counts'] / mergedSocCovid.Population) * 1000

    mergedSocCovid = pd.merge(mergedSocCovid, cases, how='inner', on = 'Zipcode')
    mergedSocCovid['Case Counts(Per 1000)'] = (mergedSocCovid['Case Counts'] / mergedSocCovid.Population) * 1000

    mergedSocCovid = mergedSocCovid.round(2)
    mergedSocCovid = mergedSocCovid.drop_duplicates()

    # mergedSocCovid.to_csv('Chicago-Covid-SocioDemographics-Cases-Deaths.csv', index=False)
    return mergedSocCovid

def prepare_data():
    # Newyork:
    # https://raw.githubusercontent.com/nychealth/coronavirus-data/master/totals/data-by-modzcta.csv
    # City of San Antonio:
    # https://cosacovid-cosagis.hub.arcgis.com/datasets/CoSAGIS::covid19-deaths-by-zip-code/about
    # Wisconsin:
    # https://data.dhsgis.wi.gov/datasets/wi-dhs::covid-19-data-by-zip-code-tabulation-area-v2/about
    ny_df = pd.read_csv('Data/Covid Data/newyork-covid19-cases-and-deaths.csv')
    ny_df = ny_df[['MODIFIED_ZCTA', 'COVID_CASE_COUNT', 'COVID_DEATH_COUNT']]
    ny_df.columns = ['Zipcode', 'Case Counts', 'Death Counts']

    sa_df = pd.read_csv('Data/Covid Data/san-antonio-covid19-cases-and-deaths.csv')
    sa_df = sa_df[['ZIP_CODE', 'Positive','Deaths']]
    sa_df.columns = ['Zipcode', 'Case Counts', 'Death Counts']

    wi_df = pd.read_csv('Data/Covid Data/wisconsin-covid19-cases-and-deaths.csv')
    wi_df = wi_df[['GEOID', 'POS_CUM_CP', 'DTH_CUM_CP']]
    wi_df.columns = ['Zipcode', 'Case Counts', 'Death Counts']

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

    sf_df = pd.read_csv('Data/Covid Data/sf_covid_data.csv')
    sf_df.columns = ['Zipcode', 'Case Counts', 'Death Counts']

    mc_df = pd.read_csv('Data/Covid Data/montgomery_county_covid_data.csv')
    mc_df.columns = ['Zipcode', 'Case Counts', 'Death Counts']

    frames = [ny_df, sa_df, wi_df, nc_df, sf_df, mc_df]
    df = pd.concat(frames)

    df['Death Counts'] = df['Death Counts'].fillna(0)
    df['Case Counts'] = df['Case Counts'].fillna(0)

    return df

def getChi():
    if(os.path.exists('Chicago-Covid-SocioDemographics-Cases-Deaths.csv')):
        chicago = pd.read_csv('Chicago-Covid-SocioDemographics-Cases-Deaths.csv')
    else:
        Cases = pd.read_csv('Data/Covid Data/COVID-19_Cases__Tests__and_Deaths_by_ZIP_Code.csv')
        Deaths = pd.read_csv("Data/Covid Data/Medical_Examiner_Case_Archive_-_COVID-19_Related_Deaths.csv")
        zip = pd.read_csv("Data/Zip_Codes.csv")
        chiZip = zip.ZIP.to_list()
        chiZip.remove(60666) #OHare Airport!!
        chiCase = wrangleCovidCaseData(Cases,chiZip)
        chiDeath = wrangleCovidDeathData(Deaths,chiZip)
        chiSoc = getSocioDem(chiZip)
        chicago = mergeChi(chiSoc, chiDeath, chiCase)
        chicago.to_csv('Chicago-Covid-SocioDemographics-Cases-Deaths.csv', index=False)

    return chicago

def getNew():
    #Wrangle additional data (New York, San Antonio, Wisconsin, North Carolina, San Francisco, Montogomery)
    if(os.path.exists('AdditionalData-Covid-SocioDemographics-Cases-Deaths.csv')):
        new = pd.read_csv('AdditionalData-Covid-SocioDemographics-Cases-Deaths.csv')
    else:
        df = prepare_data()
        newZip = df.Zipcode.tolist()
        newZip.remove('ZCTA N/A')
        newSoc = getSocioDem(newZip)
        new = pd.merge(newSoc, df, how='inner', on='Zipcode')
        zero = new[new['Population'] == 0].index.tolist()
        new = new.drop(zero)
        new = new.drop(new[new['Median household income (USD)'] == 0].index.tolist())
        new['Death Counts(Per 1000)'] = (new['Death Counts'] / new['Population']) * 1000
        new['Case Counts(Per 1000)'] = (new['Case Counts'] / new['Population']) * 1000
        new.to_csv('AdditionalData-Covid-SocioDemographics-Cases-Deaths.csv', index=False)

    return new

def getAllData(chi, new):
    if(os.path.exists('AllData-Covid-SocioDemographics-Cases-Deaths.csv')):
        all = pd.read_csv('AllData-Covid-SocioDemographics-Cases-Deaths.csv')
    else:
        chi = getChi()
        new = getNew()
        frames = [chi, new]
        all = pd.concat(frames)
        all.to_csv('AllData-Covid-SocioDemographics-Cases-Deaths.csv', index=False)
    
    return all

