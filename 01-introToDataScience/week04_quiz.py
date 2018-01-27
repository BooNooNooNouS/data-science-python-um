import numpy as np
import pandas as pd


rootdir =  '/Users/karlan/personal/dataScienceCoursera/01-introToDataScience/course1_downloads/coursera-data-science/'
filename= 'City_Zhvi_AllHomes.csv'


# ### Load gdp variable ###
# GDP = pd.read_csv(filename, skiprows=4,
#                   usecols=['Country Name', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'])

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


def get_list_of_university_towns():
    import codecs
    towns = pd.DataFrame(columns=["State", "RegionName"])

    with codecs.open(rootdir + 'university_towns.txt', encoding='utf-8') as file:
        line = file.readline()

        state_separator = -1
        state = ''
        region_counter = 0
        while line:
            # if line[-1] == ':':
            #     continue
            state_separator = line.find("[edit]")
            if state_separator >= 0:
                state = line[:state_separator]
            else:
                region = line
                if "(" in region:
                    region = region.split("(")[0]
                towns.loc[region_counter] = [state, region.strip()]
                # print("Added [{}, {}]".format(state, region))
                region_counter += 1
            line = file.readline()

    return towns

def get_gdp_data():

    gdp = pd.read_excel(rootdir + 'gdplev.xls',
                           sheet_name='Sheet1',
                           skiprows=219,
                           skip_footer=38,
                           usecols=[4, 5],
                           names=['quarter', 'gdp'])
    return gdp


def get_zillow_data():
    import math
    zillow = pd.read_csv(rootdir + 'City_Zhvi_AllHomes.csv')
    start_year = 1996
    start_month = 4
    end_year = 2016
    end_month = 6 # technically it stops at 8

    current_year = start_year
    current_month_start = start_month
    while current_year < end_year or (current_year == end_year and current_month_start < end_month):
        current_month_end = current_month_start + 3
        if current_year == end_year and current_month_end > end_month:
            current_month_end = end_month
        q_index = math.floor(current_month_start / 3) + 1
        colname = "{}q{}".format(current_year, q_index)
        cols = ["{}-{:02d}".format(current_year, m) for m in range(current_month_start, current_month_end)]
        zillow[colname] = zillow[cols].sum(axis='columns')
        #print("{}: {}".format(colname, cols))

        current_month_start += 3
        # reset the year
        if current_month_start >= 12:
            current_year += 1
            current_month_start = 1

    return zillow

def get_growth_analysis():
    recession = get_zillow_data()
    quarter_columns = [x for x in recession.columns if x[:2].isnumeric() and x[4] == "q"]

    # these are already ordered
    sum_gdp = [recession[quarter].sum() for quarter in quarter_columns]

    previous_idx = 0

    # list of GT, LT, EQ evaluating whether the current value is less than or greater than the previous one
    growth_analysis = ["NA"]
    for idx in range(1, len(quarter_columns)):
        if sum_gdp[idx] > sum_gdp[idx - 1]:
            growth_analysis.append("GROWTH")
        elif sum_gdp[idx] < sum_gdp[idx - 1]:
            growth_analysis.append("DECLINE")
        else:
            growth_analysis.append("NA")

    return pd.DataFrame(data={'quarter':quarter_columns, 'change': growth_analysis, 'sum_gdp': sum_gdp})



def get_recession_start():
    '''Returns the year and quarter of the recession start time as a
    string value in a format such as 2005q3'''
    # A recession is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive
    # quarters of GDP growth.

    analysis = get_growth_analysis()

    for idx in analysis.index - 1:
        if analysis.iloc[idx]['change'] == "DECLINE" and analysis.iloc[idx+1]['change'] == "DECLINE":
            return analysis.iloc[idx]['quarter']
    # it's possible a recession hasn't started yet
    return None


def get_recession_end():
    '''Returns the year and quarter of the recession end time as a
    string value in a format such as 2005q3'''
    recession_start = get_recession_start()
    analysis = get_growth_analysis()

    starting_idx = analysis[analysis['quarter'] == recession_start].index + 2
    # index objects are of type Int64Index which is basically a list
    for idx in range(starting_idx[0], len(analysis)):
        if analysis.iloc[idx]['change'] == "GROWTH" and analysis.iloc[idx+1]['change'] == "GROWTH":
            return analysis.iloc[idx+1]['quarter']
    # it's possible a recession hasn't finished yet
    return None


def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a
    string value in a format such as 2005q3'''
    # A recession bottom is the quarter within a recession which had the lowest GDP.

    analysis = get_growth_analysis()

    recession_start = get_recession_start()
    recession_end = get_recession_end()

    subset = analysis[(analysis['quarter'] >= recession_start) & (analysis['quarter'] < recession_end)]
    min_gdp = subset['sum_gdp'].min()
    return subset[subset['sum_gdp'] == min_gdp].reset_index().loc[0, 'quarter']


def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].

    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.

    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''

    import math
    recession = pd.read_csv(rootdir + 'City_Zhvi_AllHomes.csv')
    start_year = 2000
    start_month = 1
    end_year = 2016
    end_month = 8

    current_year = start_year
    current_month_start = start_month
    desired_columns = []
    while current_year < end_year or (current_year == end_year and current_month_start < end_month):
        current_month_end = current_month_start + 3
        if current_year == end_year and current_month_end > end_month:
            current_month_end = end_month
        q_index = math.floor(current_month_start / 3) + 1
        colname = "{}q{}".format(current_year, q_index)
        cols = ["{}-{:02d}".format(current_year, m) for m in range(current_month_start, current_month_end)]
        recession[colname] = recession[cols].mean(axis='columns')
        desired_columns.append(colname)
        # print("{}: {}".format(colname, cols))

        current_month_start += 3
        # reset the year
        if current_month_start >= 12:
            current_year += 1
            current_month_start = 1

    recession['State'] = recession['State'].map(states)

    segmented = recession.set_index(['State', 'RegionName'])
    segmented = segmented[desired_columns]


    return segmented


def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values,
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence.

    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    subset = convert_housing_data_to_quarters()

    from scipy import stats

    recession_start = get_recession_start()
    recession_bottom = get_recession_bottom()

    subset['ratio'] = subset[recession_bottom] - subset[recession_start]
    univ_towns_df = get_list_of_university_towns()
    univ_town_tuples = univ_towns_df.to_records(index=False).tolist()

    ss = subset.reset_index()
    ss['State'] = ss['State'].map(states)
    ss = ss.set_index(['State', 'RegionName'])

    univ_towns_subset = ss.loc[ss.index.isin(univ_town_tuples)]
    non_univ_towns_subset = ss.loc[~ss.index.isin(univ_town_tuples)]

    tt_result = stats.ttest_ind(univ_towns_subset['ratio'],
                    non_univ_towns_subset['ratio'],
                    nan_policy='omit')
    univ_mean = univ_towns_subset['ratio'].mean()
    non_univ_mean = non_univ_towns_subset['ratio'].mean()
    return(tt_result.pvalue < 0.01,
           tt_result.pvalue,
           "university town" if univ_mean < non_univ_mean else "non-university town")


def test_queston_1():
    import re
    import pandas as pd
    import numpy as np
    # list of unique states
    stateStr = """
    Ohio, Kentucky, American Samoa, Nevada, Wyoming
    ,National, Alabama, Maryland, Alaska, Utah
    ,Oregon, Montana, Illinois, Tennessee, District of Columbia
    ,Vermont, Idaho, Arkansas, Maine, Washington
    ,Hawaii, Wisconsin, Michigan, Indiana, New Jersey
    ,Arizona, Guam, Mississippi, Puerto Rico, North Carolina
    ,Texas, South Dakota, Northern Mariana Islands, Iowa, Missouri
    ,Connecticut, West Virginia, South Carolina, Louisiana, Kansas
    ,New York, Nebraska, Oklahoma, Florida, California
    ,Colorado, Pennsylvania, Delaware, New Mexico, Rhode Island
    ,Minnesota, Virgin Islands, New Hampshire, Massachusetts, Georgia
    ,North Dakota, Virginia
    """
    # list of regionName entries string length
    regNmLenStr = """
    06,08,12,10,10,04,10,08,09,09,05,06,11,06,12,09,08,10,12,06,
    06,06,08,05,09,06,05,06,10,28,06,06,09,06,08,09,10,35,09,15,
    13,10,07,21,08,07,07,07,12,06,14,07,08,16,09,10,11,09,10,06,
    11,05,06,09,10,12,06,06,11,07,08,13,07,11,05,06,06,07,10,08,
    11,08,13,12,06,04,08,10,08,07,12,05,06,09,07,10,16,10,06,12,
    08,07,06,06,06,11,14,11,07,06,06,12,08,10,11,06,10,14,04,11,
    18,07,07,08,09,06,13,11,12,10,07,12,07,04,08,09,09,13,08,10,
    16,09,10,08,06,08,12,07,11,09,07,09,06,12,06,09,07,10,09,10,
    09,06,15,05,10,09,11,12,10,10,09,13,06,09,11,06,11,09,13,37,
    06,13,06,09,49,07,11,12,09,11,11,07,12,10,06,06,09,04,09,15,
    10,12,05,09,08,09,09,07,14,06,07,16,12,09,07,09,06,32,07,08,
    08,06,10,36,09,10,09,06,09,11,09,06,10,07,14,08,07,06,10,09,
    05,11,07,06,08,07,05,07,07,04,06,05,09,04,25,06,07,08,05,08,
    06,05,11,09,07,07,06,13,09,05,16,05,10,09,08,11,06,06,06,10,
    09,07,06,07,10,05,08,07,06,08,06,30,09,07,06,11,07,12,08,09,
    16,12,11,08,06,04,10,10,15,05,11,11,09,08,06,04,10,10,07,09,
    11,08,26,07,13,05,11,03,08,07,06,05,08,13,10,08,08,08,07,07,
    09,05,04,11,11,07,06,10,11,03,04,06,06,08,08,06,10,09,05,11,
    07,09,06,12,13,09,10,11,08,07,07,08,09,10,08,10,08,56,07,12,
    07,16,08,04,10,10,10,10,07,09,08,09,09,10,07,09,09,09,12,14,
    10,29,19,07,11,12,13,13,09,10,12,12,12,08,10,07,10,07,07,08,
    08,08,09,10,09,11,09,07,09,10,11,11,10,09,09,12,09,06,08,07,
    12,09,07,07,06,06,08,06,15,08,06,06,10,10,10,07,05,10,07,11,
    09,12,10,12,04,10,05,05,04,14,07,10,09,07,11,10,10,10,11,15,
    09,14,12,09,09,07,12,04,10,10,06,10,07,28,06,10,08,09,10,10,
    10,13,12,08,10,09,09,07,09,09,07,11,11,13,08,10,07
    """

    df = get_list_of_university_towns()

    cols = ["State", "RegionName"]

    print('Shape test: ', "Passed" if df.shape ==
                                      (517, 2) else 'Failed')
    print('Index test: ',
          "Passed" if df.index.tolist() == list(range(517))
          else 'Failed')

    print('Column test: ',
          "Passed" if df.columns.tolist() == cols else 'Failed')
    print('\\n test: ',
          "Failed" if any(df[cols[0]].str.contains(
              '\n')) or any(df[cols[1]].str.contains('\n'))
          else 'Passed')
    print('Trailing whitespace test:',
          "Failed" if any(df[cols[0]].str.contains(
              '\s+$')) or any(df[cols[1]].str.contains(
              '\s+$'))
          else 'Passed')
    print('"(" test:',
          "Failed" if any(df[cols[0]].str.contains(
              '\(')) or any(df[cols[1]].str.contains(
              '\('))
          else 'Passed')
    print('"[" test:',
          "Failed" if any(df[cols[0]].str.contains(
              '\[')) or any(df[cols[1]].str.contains(
              '\]'))
          else 'Passed')

    states_vlist = [st.strip() for st in stateStr.split(',')]

    mismatchedStates = df[~df['State'].isin(
        states_vlist)].loc[:, 'State'].unique()
    print('State test: ', "Passed" if len(
        mismatchedStates) == 0 else "Failed")
    if len(mismatchedStates) > 0:
        print()
        print('The following states failed the equality test:')
        print()
        print('\n'.join(mismatchedStates))

    df['expected_length'] = [int(s.strip())
                             for s in regNmLenStr.split(',')
                             if s.strip().isdigit()]
    regDiff = df[df['RegionName'].str.len() != df['expected_length']].loc[
              :, ['RegionName', 'expected_length']]
    regDiff['actual_length'] = regDiff['RegionName'].str.len()
    print('RegionName test: ', "Passed" if len(regDiff) ==
                                           0 else ' \nMismatching regionNames\n {}'.format(regDiff))


if __name__ == "__main__":
    get_list_of_university_towns().to_csv("dataframeuniv.csv")
    test_queston_1()
    # university_towns = get_list_of_university_towns()
    # assert all(university_towns.columns.isin(["State", "RegionName"]))

    # print("get_list_of_university_towns: {}".format(get_list_of_university_towns()))
    # print("Recession started in " + get_recession_start())
    # print("And ended in " + get_recession_end())
    # print("With the recession bottom in" + get_recession_bottom())

    import numpy
    assert type(convert_housing_data_to_quarters().loc["Texas"].loc["Austin"].loc["2010q3"]) == numpy.float64

    # x = convert_housing_data_to_quarters()
    # assert all([x[:4].isnumeric() for x in x.columns])
    # assert len(x) == 10730
    # assert isinstance(x, pd.DataFrame)
    # assert len(x.columns) == 67
    # assert len(x.index.names) == 2
    # assert x.index.names[0] == "State"
    # assert x.index.names[1] == "RegionName"
    # print(run_ttest())

