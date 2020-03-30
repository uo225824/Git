import pandas as pd

animals=['Tiger','Bear','Moose']
print(pd.Series(animals))

numbers=[1,2,3]

print(pd.Series(numbers))

numbers = [1, 2, None]
print(pd.Series(numbers))

import numpy as np
print(np.nan == None)
print(np.isnan(np.nan))

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
print(s)
print(s.index)

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
print(s)

#Querying a Series

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
print(s.iloc[3])
print(s[3])

s = pd.Series([100.00, 120.00, 101.00, 3.00])

total = 0
for item in s:
    total+=item
print(total)

total = np.sum(s)
print(total)

#this creates a big series of random numbers
s = pd.Series(np.random.randint(0,1000,10000))
print(s.head())

print(len(s))



s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'
print(s)

original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'],
                                   index=['Cricket',
                                          'Cricket',
                                          'Cricket',
                                          'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)
print(all_countries)



#Dataframe



purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
print(df)

print(df.loc['Store 1', 'Cost'])
print(df.T)#traspose

print(df.loc['Store 1']['Cost'])

print(df.loc[:,['Name', 'Cost']])

print(df.drop('Store 1'))

copy_df = df.copy()
del copy_df['Name']
print(copy_df)


#Dataframe Indexing and Loading

costs = df['Cost']
costs+=2

print(df)

df = pd.read_csv('vinoblanco1.csv', index_col = 0, skiprows=1)
print(df.head())

df = pd.read_csv('vinoblanco1.csv')#leer csv
print(df.head())
print(df.columns)


for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True)

print(df.head())

only_gold = df.where(df['acidez.fija'] > 6.3)
print(only_gold.head())
print(df['acidez.fija'].count())
print(only_gold['acidez.fija'].count())

only_gold = only_gold.dropna()
print(only_gold.head())
print(df)

#change index

df['country'] = df.index
df = df.set_index('acidez.fija')
print(df.head())


