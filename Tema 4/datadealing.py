import pandas as pd
from io import StringIO

csv_data = '''
A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))

#Number of NaN
print(df.isnull().sum())

#Rows without NaN
df.dropna(axis=0)

#Colums without NaN
df.dropna(axis=1)

# only drop rows where all columns are NaN
# (returns the whole array here since we don't
# have a row with all values NaN)
df.dropna(how='all')


#rows with 4 real values
df.dropna(thresh=4)

# only drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])

#Imputing missing values
#Mean imputation

from sklearn.impute import SimpleImputer
import numpy as np
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data

#other method with pandas

a=df.fillna(df.mean())

print(imputed_data)
print(a)


#CATEGORICAL DATA


df = pd.DataFrame([ ['green', 'M', 10.1, 'class2'], ['red', 'L', 13.5, 'class1'], ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)


#Change fctor by numbers
size_mapping = {'XL': 3,'L': 2,'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df)

#back to the originals
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)



#ENCODING CLASS LABELS


class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping{'class1': 0, 'class2': 1}
df['classlabel'] = df['classlabel'].map(class_mapping)
df

df['classlabel'] = df['classlabel'].map(class_mapping)
df

#reverse

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df