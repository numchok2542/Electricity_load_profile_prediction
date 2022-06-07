## import the scikit-learn models ##
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor 

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold,cross_validate,KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,mean_absolute_error,mean_squared_error,r2_score,make_scorer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
#create a random forest with default values
rfr = RandomForestRegressor(max_features='sqrt')
path="dataset-new-28-02-65.csv"
#print(path)
data = pd.read_csv(path, parse_dates=True)
#print(data.columns)
#print("Data is")
#print(data)

#print(data)
#load the boston dataset
#data = load_boston()
#obtain input data
data['Time'] = pd.to_datetime(data['Time'],format= '%H:%M:%S' ).dt.hour
#print(data['Time'])
#obtain labels
#X=data.Time

X = data[["Date","Time"]]
#X = data[["Date","Time","Season"]]
#data[["winter","rainy","summer"]] = pd.get_dummies(data["Season"])
Y= data[["aircon","fridge","fan","phonec","laptop","airfilter","lightb"]]

'''Y1 = data[["aircon"]]
Y2 = data[["fridge"]]
Y3 = data[["fan"]]
Y4 = data[["phonec"]]
Y5 = data[["laptop"]]
Y6 = data[["airfilter"]]
Y7 = data[["lightb"]]'''
#print(X.size, Y.size)

'''X1_train, X1_test, y1_train, y1_test = train_test_split(X, Y1, test_size=0.33,random_state=None, shuffle=False)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, Y2, test_size=0.33,random_state=None, shuffle=False)
X3_train, X3_test, y3_train, y3_test = train_test_split(X, Y3, test_size=0.33,random_state=None, shuffle=False)
X4_train, X4_test, y4_train, y4_test = train_test_split(X, Y4, test_size=0.33,random_state=None, shuffle=False)
X5_train, X5_test, y5_train, y5_test = train_test_split(X, Y5, test_size=0.33,random_state=None, shuffle=False)
X6_train, X6_test, y6_train, y6_test = train_test_split(X, Y6, test_size=0.33,random_state=None, shuffle=False)
X7_train, X7_test, y7_train, y7_test = train_test_split(X, Y7, test_size=0.33,random_state=None, shuffle=False)'''

for train_idx, val_idx in KFold(10).split(X,Y):
    X_train = X.iloc[train_idx]
    y_train = Y.iloc[train_idx]
    X_test = X.iloc[val_idx]
    y_test = X.iloc[val_idx]
    rfr.fit(X_train.values,y_train)
    y_pred = rfr.predict(X_train.values)


'''
X1_train, X1_test, y1_train, y1_test = KFold(10).split(X, Y1)
X2_train, X2_test, y2_train, y2_test = KFold(10).split(X, Y2)
X3_train, X3_test, y3_train, y3_test = KFold(10).split(X, Y3)
X4_train, X4_test, y4_train, y4_test = KFold(10).split(X, Y4)
X5_train, X5_test, y5_train, y5_test = KFold(10).split(X, Y5)
X6_train, X6_test, y6_train, y6_test = KFold(10).split(X, Y6)
X7_train, X7_test, y7_train, y7_test = KFold(10).split(X, Y7)'''

#X_train=X_train[["Time","fridge","fan","phonec","laptop","airfilter","lightb"]]
#obtain input labels
#names = data.Name
#convert input features to dataframe
#dfX_train = pd.DataFrame(X_train,columns=["Time"])
#print(X_test.values[:, 0])
'''dfX1_train = pd.DataFrame(X1_train,columns=["Time",])
dfX2_train = pd.DataFrame(X2_train,columns=["Time",])
dfX3_train = pd.DataFrame(X3_train,columns=["Time",])
dfX4_train = pd.DataFrame(X4_train,columns=["Time",])
dfX5_train = pd.DataFrame(X5_train,columns=["Time",])
dfX6_train = pd.DataFrame(X6_train,columns=["Time",])
dfX7_train = pd.DataFrame(X7_train,columns=["Time",])

dfX1_test = pd.DataFrame(X1_test,columns=["Time",])
dfX2_test = pd.DataFrame(X2_test,columns=["Time",])
dfX3_test = pd.DataFrame(X3_test,columns=["Time",])
dfX4_test = pd.DataFrame(X4_test,columns=["Time",])
dfX5_test = pd.DataFrame(X5_test,columns=["Time",])
dfX6_test = pd.DataFrame(X6_test,columns=["Time",])
dfX7_test = pd.DataFrame(X7_test,columns=["Time",])'''
#print(dfX_train)
#Fit and predict
'''rfr.fit(dfX1_train.values,y1_train)
y_pred = rfr.predict(dfX1_test.values)


rfr.fit(dfX2_train.values,y2_train)
y_pred2 = rfr.predict(dfX2_test.values)


rfr.fit(dfX3_train.values,y3_train)
y_pred3 = rfr.predict(dfX3_test.values)

rfr.fit(dfX4_train.values,y4_train)
y_pred4 = rfr.predict(dfX4_test.values)
rfr.fit(dfX5_train.values,y5_train)
y_pred5 = rfr.predict(dfX5_test.values)
rfr.fit(dfX6_train.values,y6_train)
y_pred6 = rfr.predict(dfX6_test.values)
rfr.fit(dfX7_train.values,y7_train)
y_pred7 = rfr.predict(dfX7_test.values)

y_pred7 = rfr.predict(dfX7_test.values)'''
print(y_pred)
date = [datetime.strptime(f'{x[0]} {int(x[1])}:00:00', '%d-%m-%y %H:%M:%S') for x in X1_test.values]
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
#print(date)
plt.scatter(date[:24], y2_test[:24], s=10, color="blue",alpha=0.2)
plt.scatter(date[:24], y1_test[:24], s=10, color="red", alpha=0.2)
plt.scatter(date[:24], y3_test[:24], s=10, color="green",alpha=0.2)
plt.scatter(date[:24], y4_test[:24], s=10, color="pink", alpha=0.2)
plt.scatter(date[:24], y5_test[:24], s=10, color="orange",alpha=0.2)
plt.scatter(date[:24], y6_test[:24], s=10, color="brown", alpha=0.2)
plt.scatter(date[:24], y7_test[:24], s=10, color="teal", alpha=0.2)

airplt = plt.plot(date[:24],y_pred[:24],color="Blue",linewidth=1)
friplt = plt.plot(date[:24],y_pred2[:24],color="red",linewidth=1)
fanplt = plt.plot(date[:24],y_pred3[:24],color="green",linewidth=1)
phoplt = plt.plot(date[:24],y_pred4[:24],color="pink",linewidth=1)
lapplt = plt.plot(date[:24],y_pred5[:24],color="orange",linewidth=1)
filplt = plt.plot(date[:24],y_pred6[:24],color="brown",linewidth=1)
bulbplt = plt.plot(date[:24],y_pred7[:24],color="teal",linewidth=1)

legend = plt.legend(("Air conditioner", "Fridge","Fan","Phone","Laptop","filter","Light bulb"), loc="upper left", title="appliances")
plt.xlabel('Appliance')
plt.ylabel('On or off')
plt.title('Appliances')
plt.gcf().autofmt_xdate()
plt.show()


## use k fold cross validation to measure performance ##
scoring_metrics = {'mae': make_scorer(mean_absolute_error), 
                   'mse': make_scorer(mean_squared_error),
                   'r2': make_scorer(r2_score)}
dcScores        = cross_validate(rfr,dfX1_test.values,y_pred,cv=10,scoring=scoring_metrics)
print('Mean MAE: %.2f' % np.mean(dcScores['test_mae']))
print('Mean MSE: %.2f' % np.mean(dcScores['test_mse']))
print('Mean R2: %.2f' % np.mean(dcScores['test_r2']))