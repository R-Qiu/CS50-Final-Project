# CS50 Final Project Main
# Richard Qiu & Todd Chutichetpong

import numpy
import pandas as pd
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re

# load dataset // shape = (134740, 90)
df = pd.read_csv('/Users/apple/PycharmProjects/CS50-Final-Project/weather_data/NYC-CentralPark.csv')

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


# Extract valuable features

# Temperature (range, average, standard deviation)
tempRange = SOD['DAILYMaximumDryBulbTemp'] - SOD['DAILYMinimumDryBulbTemp']
tempRange = pd.Series(tempRange, name='tempRange')
tempRange.reset_index(inplace = True, drop=True)

tempAve = pd.Series(SOD['DAILYAverageDryBulbTemp'], name='tempAve')
tempAve.reset_index(inplace = True, drop=True)

# grouped_std = df.resample('D').std()
# tempSD['tempSD'] = grouped_std['HOURLYWETBULBTEMPF']

# Find Sun time in minutes
DAILYSunrise = pd.to_datetime(SOD['DAILYSunrise'], format='%H%M')
DAILYSunset = pd.to_datetime(SOD['DAILYSunset'], format='%H%M')
SunTime = DAILYSunset - DAILYSunrise
SunTime = SunTime.dt.total_seconds() / 60
SunTime = pd.Series(SunTime, name='SunTime')
SunTime.reset_index(inplace = True, drop=True)

# Visibility (average and standard deviation)
# grouped_ave =  df.groupby(df.index.day).mean()
# visAve = grouped_ave['HOURLYVISIBILITY']
# print(visAve)
# visSD = grouped_std['HOURLYVISIBILITY']
# print(visSD)
# m = df.groupby(df.index.day, as_index=False)['HOURLYVISIBILITY'].mean()
# print(m)

# Wind Speed
windSpeed = pd.Series(SOD['DAILYAverageWindSpeed'], name='windSpeed')
windSpeed.reset_index(inplace = True, drop=True)

# Amount of Precipitation
DAILYPrecip = pd.Series(SOD['DAILYPrecip'], name='precipAmount')
DAILYPrecip.replace('T', 0, inplace=True)
DAILYPrecip.reset_index(inplace = True, drop=True)

# Find Precipitation type using one hot encoding
weatherTypes = {'FG:01':[], 'FG+:02':[], 'TS:03':[], 'PL:04':[], 'GR:05':[], 'GL:06':[],
                'DU:07':[], 'HZ:08':[], 'BLSN:09':[], 'FC:10':[], 'WIND:11':[], 'BLPY:12':[], 'BR:13':[], 'DZ:14':[],
                'FZDZ:15':[], 'RA:16':[], 'FZRA:17':[], 'SN:18':[], 'UP:19':[], 'MIFG:21':[], 'FZFG:22':[]}

DAILYWeather = SOD['DAILYWeather'].tolist()
DAILYWeather = [str(i) for i in DAILYWeather]

for word in DAILYWeather:
    for key in weatherTypes:
        result = word.find(key)
        weatherTypes[key].append(1 if result != -1 else 0)

weatherEncoded = pd.DataFrame.from_dict(weatherTypes)

# Cloud Cover extraction
# Convert Skycondition data from dataframe into a list
cloudCondition = df['HOURLYSKYCONDITIONS'].tolist()

# Parse all elements into type string
cloudCondition = [str(i) for i in cloudCondition]

# Create a dictionary to store the extracted result
conditionExtracted = {'cloudCover':[]}

for condition in cloudCondition:
    # Extract the cloud cover number using regular expression on a string
    matchObj = re.search( r'\d\d\s\d*$', condition)
    # If data is not null, append the first 2 characters (i.e '02') into the dictionary
    if matchObj:
        foundString = matchObj.group()
        conditionExtracted['cloudCover'].append(foundString[:2])
    # If null, append '0'
    else:
        conditionExtracted['cloudCover'].append('0')

# Convert dict into dataframe and turn values from string into float
cloudCover = pd.DataFrame.from_dict(conditionExtracted)
cloudCover['cloudCover'] = cloudCover['cloudCover'].astype(float)

# Reset index into datetime to be able to group into daily data
cloudCover.set_index(df.index, inplace=True, drop=True)

# Find mean of daily data
cloudCover_daily = cloudCover.resample('D').mean()
cloudCover_daily.reset_index(inplace = True, drop=True)
cloudCover_daily.drop(cloudCover_daily.index[-1], inplace = True)

# Compile all features into a feature dataframe
features = pd.concat([tempRange, tempAve, SunTime, windSpeed, DAILYPrecip, weatherEncoded, cloudCover_daily], axis=1)
print(features)




# Split DailyWeather into 4 columns (max description)
# split = SOD['DAILYWeather'].str.split(" ", n = 3, expand = True)
# print(split)
# print(type(split))

# # Fill all None with Nan
# split.fillna(value=pd.np.nan, inplace=True)
#
# category1 = pd.get_dummies(split[0])
# category2 = pd.get_dummies(split[1])
# category3 = pd.get_dummies(split[2])
# category4 = pd.get_dummies(split[3])
#
# print(category1)
# print(category2)
# category1.join(category2, all = True)
# print(category1)
#
# # first combine the splits into one long list
# combine = pd.concat([split[0], split[1], split[2], split[3]], axis=0)
# combineDF = pd.DataFrame({combine.index, combine})
# # combine.reset_index(inplace=True, drop=True)
#
# print(type(combine))
# category = pd.get_dummies()
# print(category)

# # Integer Encode the long list
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit(combine.astype(str))
# print(label_encoder.classes_)
#
# # Binary encode the long list
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# # Split back into 5 different arrays and sum them to get one hot encoding
# result = numpy.split(onehot_encoded, 5)
# finalResult = result[0] + result[1] + result[2] + result[3] + result[4]
#
# # Delete Nan Column (last column)
# finalResult = numpy.delete(finalResult, -1, axis=1)
# print(finalResult)
# print(finalResult.shape)


# new_result = numpy.add(result[0], result[1], result[2], result[3], result[4])

# combine = pd.concat([split[0], split[1], split[2], split[3], split[4]])
# label_encoder = LabelEncoder()
# SOD[] = label_encoder.fit_transform(combine)

# label encode the combined series to ensure eveything is encoded
# one hot transform each split individually
# add the one hot encoded together
# 0 1 0 1 0...etc. (sample)





#
# onehot = pd.get_dummies(SOD['DAILYWeather'], prefix=['type'])
# print(onehot)

# selected = grouped[['DAILYWeather','DAILYPrecip', 'DAILYAverageDryBulbTemp',
#                     ]]









