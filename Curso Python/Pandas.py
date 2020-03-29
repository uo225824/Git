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