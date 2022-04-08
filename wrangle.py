import numpy as np
import pandas as pd
import datetime
import re
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

class Wrangle:

    covid = pd.read_csv("Data/Covid Data/Medical_Examiner_Case_Archive_-_COVID-19_Related_Deaths.csv")
    zip = pd.read_csv("Data/Zip_Codes.csv")
    covidCases = pd.read_csv('Data/Covid Data/COVID-19_Cases__Tests__and_Deaths_by_ZIP_Code.csv')
    zipcodes = zip.ZIP.to_list()
    zipcodes.remove(60666) #OHare Airport!!

    def scrape(self, z):
        z = str(z)
        source = requests.get('https://censusreporter.org/profiles/86000US{zip}-{zip}/'.format(zip=z)).text
        soup = BeautifulSoup(source, 'lxml')
        s = soup.findAll('script',type="text/javascript")[1].getText()
        return s

    def parse(self, param, sp):
        param = '"'+param+'", '
        try:
            var = float(re.findall(param+'"values": {"this":\s*([+-]?[0-9]+\.[0-9]+)',sp)[0])
        except:
            var = 0
        
        return var

    def getSocioDem(self):
        df = pd.DataFrame(columns = [ 'Zipcode', 'Population', 'Median age', 'Under 18(%)', '18 to 64(%)', '65 and over(%)', 'Male(%)', 'Female(%)', 'White(%)', "Black(%)", "Native(%)", "Asian(%)", "Islander(%)", "Two plus(%)", "Hispanic(%)", "Per capita income (USD)", "Median household income (USD)", "Below poverty line(%)",
                                    'Mean travel time to work (Minutes)', 'Drove Alone (%)', 'Carpooled (%)', 'Public Transit (%)', 'Bicycle (%)', 'Walked (%)', 'Other (%)', 'Worked at home (%)', 'Number of households', 'Persons per household', 'Married (%)', 'Single (%)',
                                    'Number of housing units', 'Occupied housing (%)', 'Vacant housing (%)', 'Owner Occupied (%)', 'Renter Occupied (%)', 'Median housing value',
                                    'Moved Since Prev Year(%)', 'Same House Prev Year(%)', 'No Degree(%)', 'High School(%)', 'Some College(%)', "Bachelor's(%)", "Post-grad(%)", 'Foriegn Born Population(%)', 'Europe(%)', 'Asia(%)', 'Africa(%)', 'Oceania(%)', 'Latin America(%)', 'North America(%)'],
                                    index = list(range(0,60)))

        i = 0
        for zip in self.zipcodes:
            s = self.scrape(zip)

            var = '"full_geoid": "86000US{z}", "total_population":'.format(z=str(zip))
            population = int(re.findall(var+'\s*([+-]?[0-9]+)',s)[0])
            
            median_age = self.parse("Median age",s)
            percent_under18 = self.parse("Under 18",s)
            percent_18to64 = self.parse("18 to 64",s)
            percent_65andOver = self.parse("65 and over",s)

            percent_male = self.parse("Male",s)
            percent_female = self.parse("Female",s)

            percent_white = self.parse("White",s)
            percent_black = self.parse("Black",s)
            percent_native = self.parse("Native",s)
            percent_asian = self.parse("Asian",s)
            percent_islander = self.parse("Islander",s)
            pecent_two_plus = self.parse("Two\+",s)
            percent_hispanic = self.parse("Hispanic",s)

            per_capita = self.parse("Per capita income", s)
            median_household_income = self.parse("Median household income",s)

            percent_below_poverty = self.parse("Persons below poverty line",s)

            mean_travel_time = self.parse("Mean travel time to work",s)
            drove_alone = self.parse("Drove alone",s)
            carpooled = self.parse("Carpooled",s)
            public_transit = self.parse("Public transit",s)
            bicycle = self.parse("Bicycle",s)
            walked = self.parse("Walked",s)
            other = self.parse("Other",s)
            worked_at_home = self.parse("Worked at home",s)

            number_of_households = self.parse("Number of households",s)
            persons_per_household = self.parse("Persons per household",s)

            married = self.parse("Married",s)
            single = self.parse("Single",s)

            number_of_housing_units = self.parse("Number of housing units",s)
            occupied_housing_units = self.parse("Occupied",s)
            vacant_housing_units = self.parse("Vacant",s)
            owner_housing_units = self.parse("Owner occupied",s)
            renter_housing_units = self.parse("Renter occupied",s)

            median_value_owner_occupied = self.parse("Median value of owner-occupied housing units",s)

            moved_since_previous_year = self.parse("Moved since previous year",s)
            same_house_year_ago = self.parse("Same house year ago",s)

            no_degree = self.parse("No degree",s)
            high_school = self.parse("High school",s)
            some_college = self.parse("Some college",s)
            bachelors = self.parse("Bachelor's",s)
            post_grad = self.parse("Post-grad",s)

            foriegn_born_pop = self.parse("Foreign-born population",s)
            europe = self.parse('Europe',s)
            asia = self.parse('Asia',s)
            africa = self.parse('Africa',s)
            oceania = self.parse('Oceania',s)
            latin_america = self.parse('Latin America',s)
            north_america = self.parse('North America',s)
            
            df.iloc[i] = [zip, population, median_age, percent_under18, percent_18to64, percent_65andOver, percent_male, percent_female, percent_white, percent_black, percent_native, percent_asian, percent_islander, pecent_two_plus, percent_hispanic, per_capita, median_household_income, percent_below_poverty, mean_travel_time,
            drove_alone, carpooled, public_transit, bicycle, walked, other, worked_at_home, number_of_households, persons_per_household, married, single, number_of_housing_units, occupied_housing_units, vacant_housing_units, owner_housing_units, renter_housing_units,
            median_value_owner_occupied, moved_since_previous_year, same_house_year_ago, no_degree, high_school, some_college, bachelors, post_grad, foriegn_born_pop, europe, asia, africa, oceania, latin_america, north_america]

            i+=1
        
        return df

    def wrangleCovidDeathData(self):
        self.covid = self.covid[self.covid['Residence Zip'].notna()]
        sep = '-'
        self.covid['Residence Zip'] = self.covid['Residence Zip'].apply(lambda x: x.split(sep, 1)[0])
        self.covid['Residence Zip'] = self.covid['Residence Zip'].astype(str).astype(int)
        self.covid = self.covid[self.covid['Residence Zip'].isin(self.zipcodes)]
        self.covid['Date of Death'] = pd.to_datetime(self.covid['Date of Death'])
        self.covid['Date of Death'] = self.covid['Date of Death'].dt.date
        self.covid = self.covid[self.covid['Date of Death'] <= datetime.date(2022, 3, 5)]
        self.covid = self.covid.sort_values(by=['Date of Death'])
        self.covid = self.covid[(self.covid['Manner of Death']!='ACCIDENT') & (self.covid['Manner of Death']!='SUICIDE')]
        self.covid = self.covid[self.covid['COVID Related'] == True]
        self.covid = self.covid.rename(columns={"Residence Zip": "Zipcode"})

        self.covid = self.covid.drop(['Case Number','Date of Incident','Manner of Death','Residence City', 'Primary Cause', 'Primary Cause Line A', 'Primary Cause Line B',
            'Primary Cause Line C', 'Secondary Cause', 'Gun Related', 'COVID Related', 'Opioid Related', 'Commissioner District',
            'Incident Address', 'Incident City', 'Incident Zip Code', 'longitude',
            'latitude', 'location', 'OBJECTID', 'Chicago Ward',
            'Chicago Community Area'],axis=1)

        return self.covid

    def wrangleCovidCaseData(self):
        self.covidCases['Week Start'] = pd.to_datetime(self.covidCases['Week Start']).dt.date
        self.covidCases['Week End'] = pd.to_datetime(self.covidCases['Week End']).dt.date
        self.covidCases = self.covidCases[self.covidCases['Week End'] == datetime.date(2022, 3, 5)]
        self.covidCases['Cases - Cumulative'] = self.covidCases['Cases - Cumulative'].apply(np.int64)
        self.covidCases = self.covidCases[['ZIP Code', 'Cases - Cumulative']]
        self.covidCases = self.covidCases[self.covidCases['ZIP Code'] != 'Unknown']
        self.covidCases['ZIP Code'] = self.covidCases['ZIP Code'].astype(str).astype(int)
        self.covidCases = self.covidCases[self.covidCases['ZIP Code'].isin(self.zipcodes)]
        self.covidCases = self.covidCases.rename(columns={'ZIP Code': 'Zipcode'})
        self.covidCases = self.covidCases.rename(columns={'Cases - Cumulative': 'Case Counts'})

        return self.covidCases
    
    def rawCovidCaseData(self):
        return self.covidCases

    def MergeSocCovid(self):
        self.wrangleCovidCaseData()
        self.wrangleCovidDeathData()

        soc = self.getSocioDem()
        deathCountByZip = self.covid.groupby('Zipcode').size().to_frame().reset_index().rename(columns={0: 'Death Counts'})

        mergedSocCovid = pd.merge(soc, deathCountByZip, how='inner', on = 'Zipcode')
        mergedSocCovid['Death Counts(Per 1000)'] = (mergedSocCovid['Death Counts'] / mergedSocCovid.Population) * 1000

        mergedSocCovid = pd.merge(mergedSocCovid, self.covidCases, how='inner', on = 'Zipcode')
        mergedSocCovid['Case Counts(Per 1000)'] = (mergedSocCovid['Case Counts'] / mergedSocCovid.Population) * 1000

        mergedSocCovid = mergedSocCovid.round(2)
        mergedSocCovid = mergedSocCovid.drop_duplicates()
        
        mergedSocCovid.to_csv('Chicago-Covid-SocioDemographics-Cases-Deaths.csv', index=False)
        return mergedSocCovid
        
    