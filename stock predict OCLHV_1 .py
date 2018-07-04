from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import arange
from numpy import array
from keras import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot
from matplotlib.gridspec import GridSpec
import mpl_finance as mpf


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return DataFrame(diff)


# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 5)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 5)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]

    # caculate base rmse
    base_predict = []
    base_actual = []
    for i in range(3, len(raw_values)):
        base_predict.append(raw_values[i - 1] + raw_values[i - 2] - raw_values[i - 3])
        base_actual.append(raw_values[i])
    base_predict = DataFrame(base_predict)
    base_actual = DataFrame(base_actual)
    base_predict = base_predict.values
    base_actual = base_actual.values
    for i in range(5):
        base_rmse = sqrt(mean_squared_error(base_predict[:, i], base_actual[:, i]))
        print('base rmse is :', base_rmse)
    return scaler, train, test


# fit an LSTM network to training data
def fit_lstm(train, n_vars, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_vars], train[:, n_vars:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
        print('epoch %d is finished' % i)
    return model


# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]


# make forecasts
def make_forecasts(model, n_batch, train, test, n_vars, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_vars], test[i, n_vars:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts


# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    return inverted


# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(-1, 5)
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        # inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted


# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_vars, n_seq):
    for i in range(n_seq):
        for j in range(n_vars):
            actual = [row[i][j] for row in test]
            predicted = [forecast[i][j] for forecast in forecasts]
            rmse = sqrt(mean_squared_error(actual, predicted))
            print('t+%d（%d） RMSE: %f' % ((i + 1), j, rmse))


series = read_csv('000876_OCLHV.csv', header=0, index_col=0)
# series.info()

# configure
n_lag = 1
n_seq = 3
n_vars = 5
n_test = 100
n_epochs = 10
n_batch = 1
n_neurons = 5 * 30

# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)

# fit model
model = fit_lstm(train, n_vars, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_vars, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test + 2)

actual = [row[n_vars:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test + 2)

# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_vars, n_seq)

# draw candle stick of actual and  forecasts (only next one day)
# prepare K_datas
ind = arange(len(forecasts))
forecasts = array(forecasts).reshape(len(forecasts), -1)
actual = array(actual).reshape(len(actual), -1)

# create fig subplots
fig, (ax0, ax1) = pyplot.subplots(2, sharex=True,  figsize=(15, 8))
gs = GridSpec(3, 1, hspace=0.05)

ax0 = pyplot.subplot(gs[0:2])
ax1 = pyplot.subplot(gs[2])

# draw candle stick on ax0
mpf.candlestick2_ochl(ax0, forecasts[:, 0], forecasts[:, 1], forecasts[:, 2], forecasts[:, 3], width=0.8, colorup='r',
                      colordown='g', alpha=1)
mpf.candlestick2_ochl(ax0, actual[:, 0], actual[:, 1], actual[:, 2], forecasts[:, 3], width=1, colorup='r',
                      colordown='g', alpha=0.4)
# draw volume on ax1
pyplot.bar(ind, forecasts[:, 4], width=0.8, align='center', alpha=1)
pyplot.bar(ind, actual[:, 4], width=1, align='center', alpha=0.4)
pyplot.show()
