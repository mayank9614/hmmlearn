
from __future__ import print_function
import datetime
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt, finance
from matplotlib.dates import YearLocator, MonthLocator
from hmmlearn.hmm import GaussianHMM
import math
#import pandas_datareader.data as web
ticker = "gold"
start_date = datetime.date(2010, 1, 1)
#end_date = datetime.date.today()
end_date = datetime.date.today() - datetime.timedelta(days=15)


data = pd.read_csv('data2.csv', header=0)
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df.head()
#print(df)
df.reset_index(inplace=True,drop=False)
df.head()
df.drop(['index','open','low','high','Adj Close'],axis=1,inplace=True)
#df.head()
df['date'] = df['date'].apply(datetime.datetime.toordinal)
df = list(df.itertuples(index=False, name=None))
dates = np.array([q[0] for q in df], dtype=int)
close_v = np.array([q[1] for q in df])
volume = np.array([q[2] for q in df])[1:]

diff = np.diff(close_v)
dates = dates[1:]
close_v = close_v[1:3]

X = np.column_stack([diff, volume])
plt.figure(figsize=(15, 5), dpi=100) 
plt.title(ticker + " - " + end_date.strftime("%m/%d/%Y"), fontsize = 14)
plt.gca().xaxis.set_major_locator(YearLocator())
plt.plot_date(dates,close_v,"-")
#plt.show()
print(X[0])

print("fitting to HMM and decoding ...", end="")
# Make an HMM instance and execute fit
model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000).fit(X)
# Predict the optimal sequence of internal hidden state

hidden_states = model.predict(X)
print("done")

print("Transition matrix - probability of going to any particular state")
print(model.transmat_)
print(model.predict_proba)

print("Means and vars of each hidden state")
params = pd.DataFrame(columns=('State', 'Means', 'Variance'))
for i in range(model.n_components):
    params.loc[i] = [format(i), model.means_[i],np.diag(model.covars_[i])]

fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True, figsize=(15,15))
colours = cm.rainbow(np.linspace(0, 1, model.n_components))

for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(dates[mask], close_v[mask], ".", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())

    ax.grid(True)
plt.show()

expected_returns_and_volumes = np.dot(model.transmat_, model.means_)
returns_and_volume_columnwise = list(zip(*expected_returns_and_volumes))
expected_returns = returns_and_volume_columnwise[0]
expected_volumes = returns_and_volume_columnwise[1]
params = pd.concat([pd.Series(expected_returns), pd.Series(expected_volumes)], axis=1)
params.columns= ['Returns', 'Volume']
print (params)

lastN = 7
start_date = datetime.date.today() - datetime.timedelta(days=lastN*2) #even beyond N days
end_date = datetime.date.today() 

dates = np.array([q[0] for q in df], dtype=int)

predicted_prices = []
predicted_dates = []
predicted_volumes = []
actual_volumes = []

for idx in range(lastN):
    state = hidden_states[-lastN+idx]
    current_price = df[-lastN+idx][1]
    volume = df[-lastN+idx][2]
    actual_volumes.append(volume)
    current_date = datetime.date.fromordinal(dates[-lastN+idx])
    predicted_date = current_date + datetime.timedelta(days=1)
    predicted_dates.append(predicted_date)
    predicted_prices.append(current_price + expected_returns[state])
    predicted_volumes.append(np.round(expected_volumes[state]))    

#Returns
plt.figure(figsize=(15, 5), dpi=100) 
plt.title(ticker, fontsize = 14)
plt.plot(predicted_dates,close_v[-lastN:])
plt.plot(predicted_dates,predicted_prices)
plt.legend(['Actual','Predicted'])
plt.show()

#Volumes
plt.figure(figsize=(15, 5), dpi=100) 
plt.title(ticker, fontsize = 14)
plt.plot(predicted_dates,actual_volumes)
plt.plot(predicted_dates,predicted_volumes)
plt.legend(['Actual','Predicted'])
plt.show()

Actual=close_v[-lastN:]

forecast_errors = [Actual[i]- predicted_prices[i] for i in range(len(predicted_prices))]

mean_forecast_error = statistics.mean(forecast_errors)
mean_forecast_error

bias = sum(forecast_errors) * 1.50/len(predicted_prices)
print('Bias: %f' % bias)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Actual, predicted_prices)
print('MSE: %f' % mse)


rmse  =  math.sqrt((mse))

print('RMSE: %f' % rmse)





