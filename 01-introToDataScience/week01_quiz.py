import numpy as np
import pandas as pd


def prep_data_part_1():
    df = pd.read_csv('course1_downloads/olympics.csv', index_col=0, skiprows=1)

    for col in df.columns:
        if col[:2] == '01':
            df.rename(columns={col: 'Gold' + col[4:]}, inplace=True)
        if col[:2] == '02':
            df.rename(columns={col: 'Silver' + col[4:]}, inplace=True)
        if col[:2] == '03':
            df.rename(columns={col: 'Bronze' + col[4:]}, inplace=True)
        if col[:1] == '№':
            df.rename(columns={col: '#' + col[1:]}, inplace=True)

    names_ids = df.index.str.split('\s\(')  # split the index by '('

    df.index = names_ids.str[0]  # the [0] element is the country name (new index)
    df['ID'] = names_ids.str[1].str[:3]  # the [1] element is the abbreviation or ID (take first 3 characters from that)

    df = df.drop('Totals')
    df.head()
    return df


def answer_zero(df):
    return df.iloc[0]

def answer_one(df):
    return df['Gold'].idxmax()


def answer_two(df):
    df_clone = df.assign(Delta=abs(df['Gold'] - df['Gold.1']))
    return df_clone['Delta'].idxmax()


def answer_three(df):
    df_slice = df[(df['Gold'] > 0) & (df['Gold.1'] > 0)]
    df_slice = df_slice.assign(Relative_Diff=abs(df_slice['Gold'] - df_slice['Gold.1']) /
                                             (df_slice['Gold'] + df_slice['Gold.1']))
    return df_slice['Relative_Diff'].idxmax()


def answer_four(df):
    # Question 4
    # Write a function that creates a Series called "Points" which is a weighted value where each gold medal (Gold.2)
    # counts for 3 points, silver medals (Silver.2) for 2 points, and bronze medals (Bronze.2) for 1 point. The function
    # should return only the column (a Series object) which you created, with the country names as indices.
    # This function should return a Series named Points of length 146
    df_clone = df.assign(Points=3 * df['Gold.2'] + 2 * df['Silver.2'] + df['Bronze.2'])
    return df_clone['Points']

def run_part_1():
    df = prep_data_part_1()

    print("Exercise 1")
    print("1. Which country has won the most gold medals in summer games?")
    r1 = answer_one(df)
    print("\t\t\t" + r1)
    assert isinstance(r1, str)

    print("2. Which country had the biggest difference between their summer and winter gold medal counts?")
    r2 = answer_two(df)
    print("\t\t\t" + r2)
    assert isinstance(r2, str)

    print(
        "3. Which country has the biggest difference between their summer gold medal counts and winter gold medal counts relative to their total gold medal count?")
    r3 = answer_three(df)
    print("\t\t\t" + r3)
    assert isinstance(r3, str)

    r4 = answer_four(df)
    r4.head()
    assert isinstance(r4, pd.Series)
    assert r4.size == 146


def answer_five(census_df):
    census_min = census_df.loc[census_df['SUMLEV'] == 50, ['SUMLEV', 'STNAME']]
    groupped = census_min.groupby('STNAME').agg('count')
    return groupped['SUMLEV'].idxmax()



def answer_six(census_df):
    census_states = census_df.loc[census_df['SUMLEV'] == 50, ['STNAME', 'CTYNAME', 'CENSUS2010POP']]
    top3perstate = census_states.groupby('STNAME')['CENSUS2010POP'].nlargest(3)
    summarized = top3perstate.groupby(level=0).sum().reset_index()

    return summarized.sort_values('CENSUS2010POP', ascending=False).head(3)['STNAME'].tolist()


def answer_seven(census_df):

    pops = ['POPESTIMATE2010', 'POPESTIMATE2011', 'POPESTIMATE2012', 'POPESTIMATE2013', 'POPESTIMATE2014', 'POPESTIMATE2015']
    census_county = census_df.loc[census_df['SUMLEV'] == 50, ['CTYNAME'] + pops]
    census_county = census_county.assign(MaxPop=census_county[pops].max(axis=1))
    census_county = census_county.assign(MinPop=census_county[pops].min(axis=1))
    census_county = census_county.assign(AbsChange=census_county['MaxPop']-census_county['MinPop'])
    census_county = census_county.set_index('CTYNAME')

    return census_county['AbsChange'].idxmax()


def answer_eight(census_df):
    # only counties, with regions 1 or 2, whose name starts with Washington, and where POPESTIMATE2015 > POPESTIMATE2014

    condition = (census_df['SUMLEV'] == 50) & \
                (census_df['REGION'] < 3) & \
                (census_df['CTYNAME'].str.startswith('Washington')) & \
                (census_df['POPESTIMATE2015'] > census_df['POPESTIMATE2014'])

    desired_columns = ['STNAME', 'CTYNAME']
    filtered = census_df.loc[condition, desired_columns]
    return filtered.sort_index()




def run_part_2():
    census_df = pd.read_csv('course1_downloads/census.csv')

    print("Exercise 2")

    print("5. Which state has the most counties in it?")
    r5 = answer_five(census_df)
    print("\t\t\t" + r5)
    assert isinstance(r5, str)

    print("6. Only looking at the three most populous counties for each state, what are the three most populous states (in order of highest population to lowest population)?")
    r5 = answer_six(census_df)
    print("\t\t\t{}".format(r5))
    assert isinstance(r5, list)
    assert isinstance(r5[0], str)

    print("7. Which county has had the largest absolute change in population within the period 2010-2015?")
    r5 = answer_seven(census_df)
    print("\t\t\t" + r5)
    assert isinstance(r5, str)


if __name__ == "__main__":
    # run_part_1()
    run_part_2()

    print('EOF')
