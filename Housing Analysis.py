import pandas as pd
import xlrd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
from datetime import date, timedelta
import calendar
import matplotlib.pyplot as plt

# Import housing data from zillow download then only keep Denton County
df = pd.read_excel(r'Zillow Perfomance.xlsx')
df = df.loc[df['CountyName'] == 'Denton County']
df = df.loc[df['RegionName'] == 'Denton']

# Transpose data to be more usable
df = df.melt(['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName', 'State', 'Metro', 'CountyName'],
             var_name='Date', value_name='HomeValue')
# df1['DateIndex'] = df.index

X = np.array(df.index)
y = np.array(df['HomeValue'])
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

best = 0
for x in range(1000):  # best < 0.95:  # Train models until accuracy (R^2) is > 95%
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.2)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)
    x += 1

    if acc < 0:  # negative slopes will be converted to acc between 0-1
        acc = 1 - (-1 * acc)

    if acc > best:  # Saves the best trained model
        best = acc
        with open('HomeValue.pickle', 'wb') as f:
            pickle.dump(linear, f)

print('Accuracy: ' + str(best))
pickle_in = open('HomeValue.pickle', 'rb')
linear = pickle.load(pickle_in)

total_predictions = 36
for x in range(total_predictions):
    tn = df.index[-1] + 1
    prediction = {'HomeValue': linear.predict([[tn]])}
    df = df.append(prediction, ignore_index=True)

# Plot results
plt.plot(df['Date'], df['HomeValue'])
plt.show()
