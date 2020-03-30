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