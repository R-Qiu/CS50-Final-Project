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
windSpeed.fillna(0, inplace=True)
windSpeed.reset_index(inplace = True, drop=True)

# Amount of Precipitation
DAILYPrecip = pd.Series(SOD['DAILYPrecip'], name='precipAmount')
DAILYPrecip.replace('T', 0, inplace=True)
DAILYPrecip.fillna(0, inplace=True)
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
features.apply(pd.to_numeric)

from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

# Load Google trends data as labels
gtrend = pd.read_csv('/Users/apple/PycharmProjects/CS50-Final-Project/trends_data/NYC_gtrends_2008-2017.csv')
gtrend['date'] = pd.to_datetime(gtrend['date'])
gtrend = gtrend.loc[gtrend['date'] <= '2017-12-31']
gtrend = gtrend['hits']
gtrend.apply(pd.to_numeric)

# Split training (8 years) and testing data (last 2 years)
X = features.iloc[:2922]
X_test = features.tail(730)
X = X.values
X_test = X_test.values

Y = gtrend.iloc[:2922]
Y_test = gtrend.tail(730)
Y = Y.values
Y_test = Y_test.values


# create model
model = Sequential()
model.add(Dense(10, input_dim=27, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

mcp = ModelCheckpoint(filepath="model_fold_weights.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Fit the model
history = model.fit(X, Y, validation_split=0.20, nb_epoch=40, batch_size=10, callbacks=[mcp], verbose=0)

# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test' ], loc= 'upper left')
plt.show()

# load weights
model.load_weights("model_fold_weights.h5")
# Compiling the Model:
model.compile(loss='mean_squared_error', optimizer='adam')
prediction = model.predict(X_test)

r2 = r2_score(Y_test, prediction)
MSE = mean_squared_error(Y_test, prediction)
exp_var = explained_variance_score(Y_test, prediction)
print(r2)
print(MSE)
print(exp_var)

plt.plot(prediction)
plt.plot(Y_test)
plt.show()


# # define 5-fold cross validation test
# kfold = KFold(n_splits=5)
#
# i = 0
# for train, validate in kfold.split(X):
#     X_train = X.iloc[train]
#     X_validate = X.iloc[validate]
#
#     Y_train = Y.iloc[train]
#     Y_validate = Y.iloc[validate]
#
#     # Creating the Neural Network:
#     model = Sequential()
#     model.add(Dense(5, input_dim=5, kernel_initializer='normal'))
#     model.add(Activation('relu'))
#
#     # model.add(Dense(3, kernel_initializer='normal'))
#     # model.add(Activation('relu'))
#
#     model.add(Dense(1, kernel_initializer='uniform'))
#
#     # Compiling the Model:
#     model.compile(loss='mean_squared_error', optimizer='adam')
#
#     # Creating Callback function:
#     mcp = ModelCheckpoint(filepath="model_fold_weights.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)
#     # es = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='auto')
#
#     # Fit the model
#     history = model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), nb_epoch=100, batch_size=20,
#                         callbacks=[mcp], verbose=1)
#
#     # # summarize history for accuracy
#     # plt.plot(history.history['loss'])
#     # plt.plot(history.history['val_loss'])
#     # plt.show()
#
#     # load weights
#     model.load_weights("model_fold_weights.h5")
#     # Compiling the Model:
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     prediction = model.predict(X_validate)
#
#     print('fold: %d' % i)
#     print(prediction)
#     print(Y_validate)
#
#     i += 1

    # # Compute ROC curve and area the curve
    # fpr, tpr, thresholds = roc_curve(Y[test], prediction[:, 0])
    # tprs.append(interp(mean_fpr, fpr, tpr))
    # tprs[-1][0] = 0.0
    # roc_auc = auc(fpr, tpr)
    # aucs.append(roc_auc)
    # plt.plot(fpr, tpr, lw=1, alpha=0.3,
    #          label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

