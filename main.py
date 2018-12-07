# CS50 Final Project Main
# Richard Qiu & Todd Chutichetpong

import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

###########################################################################
# Extract valuable features
###########################################################################

# Load dataset // shape = (134740, 90)
df = pd.read_csv('./weather_data/LA-Downtown.csv')

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

# Temperature (range, average, standard deviation)
tempRange = SOD['DAILYMaximumDryBulbTemp'] - SOD['DAILYMinimumDryBulbTemp']
tempRange = pd.Series(tempRange, name='tempRange')
tempRange.reset_index(inplace = True, drop=True)

tempAve = pd.Series(SOD['DAILYAverageDryBulbTemp'], name='tempAve')
tempAve.reset_index(inplace = True, drop=True)

# Find Sun time in minutes
DAILYSunrise = pd.to_datetime(SOD['DAILYSunrise'], format='%H%M')
DAILYSunset = pd.to_datetime(SOD['DAILYSunset'], format='%H%M')
SunTime = DAILYSunset - DAILYSunrise
SunTime = SunTime.dt.total_seconds() / 60
SunTime = pd.Series(SunTime, name='SunTime')
SunTime.reset_index(inplace = True, drop=True)

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


###########################################################################
# Implementing Neural Network Model
###########################################################################

# Load Google trends data as labels
gtrend = pd.read_csv('./trends_data/la_mvavg_depression_gtrends_2008-2017.csv')
gtrend['date'] = pd.to_datetime(gtrend['date'])
gtrend = gtrend.loc[ gtrend['date'] <= '2017-12-31']
gtrend = gtrend['moving_average']
gtrend.apply(pd.to_numeric)

# Split training (8 years) and testing data (last 2 years)
X = features.iloc[3:2922]
X_test = features.tail(730)
X = X.values
X_test = X_test.values

Y = gtrend.iloc[3:2922]
Y_test = gtrend.tail(730)
Y = Y.values
Y_test = Y_test.values

# Create model
model = Sequential()
model.add(Dense(10, input_dim=27, kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10, kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1, kernel_initializer='normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
mcp = ModelCheckpoint(filepath="model_fold_weights.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Fit the model
history = model.fit(X, Y, validation_split=0.20, nb_epoch=50, batch_size=50, callbacks=[mcp], verbose=0)

# Plotting history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation' ], loc= 'upper left')
plt.show()

# Load weights
model.load_weights("model_fold_weights.h5")

# Compiling the Model:
model.compile(loss='mean_squared_error', optimizer='adam')

# Use model and its weights to predict
prediction = model.predict(X_test)

# Evaluate using MSE
MSE = mean_squared_error(Y_test, prediction)
print(MSE)

# Plot prediction vs. truth
plt.plot(prediction)
plt.plot(Y_test)
plt.ylabel('Normalized google hits')
plt.xlabel('Time (days)')
plt.title('Model Prediction')
plt.legend(['prediction', 'truth' ], loc= 'upper left')
plt.show()
