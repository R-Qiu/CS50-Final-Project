# CS50 Final Project Main
# Richard Qiu & Todd Chutichetpong

import numpy
import pandas as pd
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# load dataset // shape = (134740, 90)
df = pd.read_csv('/Users/apple/PycharmProjects/CS50-Final-Project/NYC-CentralPark.csv')

# Filter out FM-16
df = df.loc[df['REPORTTPYE'] != 'FM-16']

# Select SOD
SOD = df.loc[df['REPORTTPYE'] == 'SOD']

# Group into daily data
new = df['DATE'].str.split(" ", n = 1, expand = True)
df['date'] = new[0]
df['time'] = new[1]
df.drop(columns = ['DATE'], inplace = True)

df.set_index(df['date'], inplace = True)
df.index = pd.to_datetime(df.index)

# grouped = df.resample('D').sum()
# print(grouped.head(10))


# Extract valuable features

# Find Temp range
SOD['TempRange'] = SOD['DAILYMaximumDryBulbTemp'] - SOD['DAILYMinimumDryBulbTemp']

# Find Temp sd
grouped = df.resample('D').std()
print(grouped['HOURLYWETBULBTEMPF'])
print(grouped['HOURLYDRYBULBTEMPF'])

# Find Sun time
SOD['DAILYSunrise'] = pd.to_datetime(SOD['DAILYSunrise'], format='%H%M')
SOD['DAILYSunset'] = pd.to_datetime(SOD['DAILYSunset'], format='%H%M')
SOD['SunTime'] = SOD['DAILYSunset'] - SOD['DAILYSunrise']

# Find Precipitation type using one hot encoding
label = ['FG:01', 'FG+:02', 'TS:03', 'PL:04', 'GR:05', 'GL:06', 'DU:07', 'HZ:08', 'BLSN:09', 'FC:10',
        'WIND:11', 'BLPY:12', 'BR:13', 'DZ:14', 'FZDZ:15', 'RA:16', 'FZRA:17', 'SN:18', 'UP:19', 'MIFG:21',
        'FZFG:22']
values = array(label)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

def foo(name):
    for email in list_of_emails:
        if name.lower() in email:
            return email

df['Email'] = df['Name'].apply(foo)

# split = SOD['DAILYWeather'].str.split(" ", n = 1, expand = True)
# SOD['DAILYWeather']

onehot = pd.get_dummies(SOD['DAILYWeather'], prefix=['type'])
print(onehot)

# selected = grouped[['DAILYWeather','DAILYPrecip', 'DAILYAverageDryBulbTemp',
#                     ]]









