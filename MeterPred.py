# import necessary library
import math
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import to_numeric
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

# read raw dataset from csv file
df = pd.read_csv('minutepower.csv')

# Date data preparation convert 01-02-22 into Day[01] Month[02] and Year[22]
date_raw = df['Date']
Day = []
Month = []
Year = []

for date in date_raw:
    d, m, y = date.split('/')
    Day.append(to_numeric(d)), Month.append(to_numeric(m)), Year.append(to_numeric(y))

# convert list into DataFrame
df_date = pd.DataFrame({'Day':Day, 'Month':Month, 'Year':Year}, columns=['Day','Month','Year'])

# Time data preparation convert 00:01:02 into hour[00] minute[01] and second[02]
time_raw = df['Time']
hour = []
minute = []
second = []

for time in time_raw:
    h, m, s = time.split(':')
    hour.append(to_numeric(h)), minute.append(to_numeric(m)), second.append(to_numeric(s))

df_time = pd.DataFrame({'Hour':hour}, columns=['Hour'])

# Season data preparation convert (summer, rainy, winter) into (1,2,3)
season_raw = df['Season']
season = []

for s in season_raw:
    if s == 'summer':
        season.append(1)
    elif s == 'rainy':
        season.append(2)
    elif s == 'winter':
        season.append(3)
    else:
        continue

df_season = pd.DataFrame({'Season':season}, columns=['Season'])

# drop old columns
df = df.drop('Date', axis=1)
df = df.drop('Time', axis=1)
df = df.drop('Season', axis=1)
df = pd.concat([df_date, df_time, df_season, df], axis=1)

# considering scaling data...
X = df.iloc[:,:6]
y = df.iloc[:,6:]

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size = 0.4, random_state = 1)

regr = RandomForestRegressor(max_depth=7, n_estimators=100, random_state=1)
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)
r2_randomforest = r2_score(y_test, y_pred)

# test prediction Date = 03, Month = 01, Year = 22, Hour = 22, Season = 3, Watts = 2852
regr_predict = regr.predict([[3, 1, 22, 22, 3, 2852]])

#print(regr_predict)
print(r2_randomforest)

filename = "rdfr.sav"
pickle.dump(regr, open(filename, 'wb'))

# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, y_test)
# result

# using neural network with 2 dense layers
# from keras.models import Sequential
# from keras.layers import Dense, BatchNormalization, Dropout

# model = Sequential()
# model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Dense(8, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.1))
# model.add(Dense(y_train.shape[1], activation='sigmoid'))
# model.add(BatchNormalization())
# model.add(Dropout(0.05))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

# history = model.fit(X_train,y_train, epochs=375, verbose=1, validation_data=(X_test, y_test))
# neural_predict = model.predict([[3, 1, 22, 22, 3, 2852]])

# model.save("nn.h5")