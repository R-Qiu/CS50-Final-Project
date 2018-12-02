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

# Drop the Nan extra row
cloudCover_daily.drop(cloudCover_daily.index[-1], inplace = True)

# Compile all features into a feature dataframe
features = pd.concat([tempRange, tempAve, SunTime, windSpeed, DAILYPrecip, weatherEncoded, cloudCover_daily], axis=1)


from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Split training (8 years) and testing data (last 2 years)
X_train = features.iloc[:2922]
X_test = features.tail(730)

# define 5-fold cross validation test
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=250)

i = 0
for train, test in kfold.split(X, Y):
    # Creating the Neural Network:
    model = Sequential()
    model.add(Dense(5, input_dim=27, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(5, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Compiling the Model:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Creating Callback function:
    mcp = ModelCheckpoint(filepath="final_model_fold" + str(i) + "_weights.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    # es = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='auto')

    # Fit the model
    history = model.fit(X_train_final, Y[train], validation_data=(X_test_final, Y[test]), nb_epoch=200, batch_size=3,
                        callbacks=[mcp], verbose=0)

    # load weights
    model.load_weights("final_model_fold" + str(i) + "_weights.h5")
    # Compiling the Model:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    prediction = model.predict(X_test_final)

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# evaluate model with standardized dataset
estimators = []
estimators.append(( standardize , StandardScaler()))
estimators.append(( mlp , KerasRegressor(build_fn=baseline_model, nb_epoch=50,
batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)

