import pandas as pd
import numpy as np

energy_filename = '/Users/karlan/personal/dataScienceCoursera/01-introToDataScience/course1_downloads/coursera-data-science/Energy Indicators.xls'
gdp_filename = '/Users/karlan/personal/dataScienceCoursera/01-introToDataScience/course1_downloads/coursera-data-science/world_bank.csv'
scimen_filename = '/Users/karlan/personal/dataScienceCoursera/01-introToDataScience/course1_downloads/coursera-data-science/scimagojr-3.xlsx'

energy_filename = 'Energy Indicators.xls'
gdp_filename = 'world_bank.csv'
scimen_filename = 'scimagojr-3.xlsx'

#### Load energy variable ###
energy = pd.read_excel(energy_filename,
                       sheet_name='Energy',
                       skiprows=17,
                       skip_footer=38,
                       usecols=list(range(2, 7)),
                       names=['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable'])
# convert energy supply to gigajoules
for colname in ['Energy Supply', 'Energy Supply per Capita']:
    energy[colname] = pd.to_numeric(energy[colname], errors='coerce')

energy['Energy Supply'] *= 1000000
# remove parenthesis with everything inside, as well as numbers
energy['Country'] = energy['Country'].str.replace(r'(?:\(.*\)|\d)', '').str.strip()
energy = energy.replace({'Country':
                             {'Republic of Korea': 'South Korea',
                              'United States of America': 'United States',
                              'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
                              'China, Hong Kong Special Administrative Region': 'Hong Kong'}})


### Load gdp variable ###
GDP = pd.read_csv(gdp_filename, skiprows=4,
                  usecols=['Country Name', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'])
GDP = GDP.replace({'Country Name':
                       {'Korea, Rep.': 'South Korea',
                        'Iran, Islamic Rep.': 'Iran',
                        'Hong Kong SAR, China': 'Hong Kong'}}).rename(columns={'Country Name': 'Country'})


### Load scimen variable ###

ScimEn = pd.read_excel(scimen_filename)

def answer_one():
    scimEnSubset = ScimEn[ScimEn['Rank'] < 16]

    gdp_scim = pd.merge(GDP, scimEnSubset, how='inner', on='Country')
    gdp_scim_en = pd.merge(gdp_scim, energy, how='inner', on='Country')
    return gdp_scim_en.set_index('Country')

def answer_two():
    outer_merge = pd.merge(
        pd.merge(ScimEn, energy, how='outer', on='Country'),
            GDP, how='outer', on='Country')

    inner_merge = pd.merge(
        pd.merge(ScimEn, energy, how='inner', on='Country'),
        GDP, how='inner', on='Country')

    return len(outer_merge) - len(inner_merge)


def answer_three():
    Top15 = answer_one()
    avgGDP = (Top15[[str(x) for x in range(2006, 2016)]].mean(axis=1)) \
        .sort_values(ascending=False)

    return avgGDP


def answer_four():
    Top15 = answer_one()
    sixthTop = answer_three().index[5]
    return Top15.loc[sixthTop, '2015'] - Top15.loc[sixthTop, '2006']


def answer_five():
    Top15 = answer_one()
    return Top15['Energy Supply per Capita'].mean()


def answer_six():
    Top15 = answer_one()
    countryWithMax = Top15['% Renewable'].idxmax()
    return countryWithMax, Top15.loc[countryWithMax, '% Renewable']


def answer_seven():
    Top15 = answer_one()
    citationsRatio = Top15['Self-citations'] / Top15['Citations']
    countryWithMax = citationsRatio.idxmax()
    return countryWithMax, citationsRatio[countryWithMax]


def answer_eight():
    Top15 = answer_one()
    population_est = (Top15['Energy Supply'] / Top15['Energy Supply per Capita']).sort_values(ascending=False)
    third_series = population_est[population_est == population_est.iloc[2]]
    return third_series.index[0]


def answer_nine():
    Top15 = answer_one()
    Top15['Citable documents per Capita'] = Top15['Citable documents'] * Top15['Energy Supply per Capita'] / Top15[
        'Energy Supply']
    return Top15['Citable documents per Capita'].corr(Top15['Energy Supply per Capita'])


def answer_ten():
    Top15 = answer_one()
    median = (Top15['% Renewable']).median()
    Top15['Above Median'] = (Top15['% Renewable'] >= median).astype(int)
    highRenew = Top15.sort_values(['Above Median', 'Rank'], ascending=[False, True])
    return highRenew['Above Median']


ContinentDict = {'China': 'Asia',
                 'United States': 'North America',
                 'Japan': 'Asia',
                 'United Kingdom': 'Europe',
                 'Russian Federation': 'Europe',
                 'Canada': 'North America',
                 'Germany': 'Europe',
                 'India': 'Asia',
                 'France': 'Europe',
                 'South Korea': 'Asia',
                 'Italy': 'Europe',
                 'Spain': 'Europe',
                 'Iran': 'Asia',
                 'Australia': 'Australia',
                 'Brazil': 'South America'}


def answer_eleven():
    Top15 = answer_one()
    Top15['Estimated Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    result = Top15['Estimated Population']\
        .groupby(ContinentDict)\
        .agg(['count', 'sum', 'mean', 'std'])\
        .rename(columns={'count': 'size'})

    result.index.rename('Continent', inplace=True)

    return result


def answer_twelve():
    Top15 = answer_one()
    Top15['Renewable bin'] = pd.cut(Top15['% Renewable'], 5)
    Top15['Continent'] = Top15.index.map(lambda x: ContinentDict.get(x))
    return Top15.groupby(['Continent', 'Renewable bin'])['Renewable bin'].agg('count').dropna()


def answer_thirteen():
    Top15 = answer_one()
    return (Top15['Energy Supply'] / Top15['Energy Supply per Capita'])\
        .map(lambda x: "{:,}".format(x))

if __name__ == "__main__":
    print("answer_one: {}".format(answer_one().shape))
    print("answer_two: {}".format(answer_two()))
    print("answer_three: {}".format(answer_three()))
    print("answer_four: {}".format(answer_four()))
    print("answer_five: {}".format(answer_five()))
    print("answer_six: {}".format(answer_six()))
    print("answer_seven: {}".format(answer_seven()))
    print("answer_eight: {}".format(answer_eight()))
    print("answer_nine: {}".format(answer_nine()))
    print("answer_ten: {}".format(answer_ten()))
    print("answer_eleven: {}".format(answer_eleven()))
    print("answer_twelve: {}".format(answer_twelve()))
    print("answer_thirteen: {}".format(answer_thirteen()))
