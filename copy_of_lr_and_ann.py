
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv('Load Profile & DSM.csv')

dataset

# Initialize an empty list to store extracted values
load = []
scheduled = []
ODUD = []
# Iterate over every third column (starting from the second column)
for col_idx1 in range(1, dataset.shape[1], 3):
    # Extract values from the current column, starting from the fourth row
    column_values1 = dataset.iloc[2:, col_idx1].values.tolist()
    # Append the extracted values to the list
    load.extend(column_values1)

load = np.array(load)

print("Shape of load:", load.shape)
print(load)

for col_idx2 in range(2, dataset.shape[1], 3):
    column_values2 = dataset.iloc[2:, col_idx2].values.tolist()
    scheduled.extend(column_values2)
scheduled = np.array(scheduled)

print("Shape of scheduled:",scheduled.shape)
print(scheduled)

for col_idx3 in range(3, dataset.shape[1], 3):
    column_values3 = dataset.iloc[2:, col_idx3].values.tolist()
    ODUD.extend(column_values3)
ODUD = np.array(ODUD)

print("Shape of ODUD:",ODUD.shape)
print(ODUD)

load

data_numeric = pd.to_numeric(load, errors='coerce')
load=np.array(load, dtype=float)
data_numeric = pd.to_numeric(scheduled, errors='coerce')
scheduled=np.array(scheduled, dtype=float)
data_numeric = pd.to_numeric(ODUD, errors='coerce')
ODUD=np.array(ODUD, dtype=float)
# Check the data type after conversion
print(load.dtype)
print(scheduled)
load

load[0:96].max()

df = pd.DataFrame({'Load': load, 'Scheduled': scheduled, 'OD/UD': ODUD})
df.shape

missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)
#print(len(missing_values))

import pandas as pd

# Your original DataFrame (df) with Load, Scheduled, and OD/UD columns
# Assuming df is already defined

# Define the start and end dates
start_date = '2022-04-01 00:00:00'
end_date = '2023-03-31 00:00:00'

# Create a DatetimeIndex with 15-minute frequency
date_range = pd.date_range(start=start_date, end=end_date, freq='15T')

# Create a DataFrame with the dates
df_dates = pd.DataFrame({'timestamp': date_range})

# Change the format of the timestamp to day-month-2023
df_dates['timestamp'] = df_dates['timestamp'].dt.strftime('%d-%m-%Y %H:%M:%S')

# Insert the timestamp column at the beginning of the DataFrame
df.insert(0, 'timestamp', df_dates['timestamp'])

# Display the updated DataFrame
print(df)

# df.to_csv('data_extraction.csv', index=False)

# from google.colab import files

# output_csv_path = 'arhf.csv'

# files.download(output_csv_path)

df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M:%S')

weekday_dummies = pd.get_dummies(df['timestamp'].dt.dayofweek, prefix='weekday')

weekday_dummies.columns = ['weekday_' + str((i + 1) % 7) for i in range(7)]

df = pd.concat([df, weekday_dummies], axis=1)
#df.drop(columns=['timestamp'], inplace=True)

print(df)

df.head()

df

# #df.to_csv('Predictive_matrix.csv', index=False)

time=df.iloc[:,0].values

n = len(time) - 960

time_some = time[:n]
time_some.shape
time_remaining = time[n:]

# print("time_some shape:", time_some.shape)
# print("time_remaining shape:", time_remaining.shape)

# X = df.iloc[:, 4:].values
# y = df.iloc[:,1].values

# z = df.iloc[:,2].values

# n = len(z) - 960

# z_some = z[:n]
# z_remaining = z[n:]

# print("z_some shape:", z_some.shape)
# print("z_remaining shape:", z_remaining.shape)

# z_some

# # Get the names of columns for X
# X_columns = df.columns[4:]

# # Get the name of the column for y
# y_column = df.columns[1]

# print("Columns for X:", X_columns)
# print("Column for y:", y_column)

# X
training_set = df.iloc[:, 1].values
# training_set=training_set.reshape(-1,1)
# training_set.shape
# Feature Scaling
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range = (0, 1))
# training_set_scaled = sc.fit_transform(training_set)
training_set=training_set.reshape(-1,1)
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(7, 31776):
    X_train.append(training_set[i-7:i, 0])
    y_train.append(training_set[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train.shape
y_train.shape
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# pip install keras
# pip install tensorflow
# from tensorflow import keras

# Importing the Keras libraries and packages
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
#pip install --upgrade tensorflow keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)





m = len(time) - 960

load_train = load[:m]
load_test = load[m:]

print("Load_train shape:",load_train.shape)
print("Load_test shape:", load_test.shape)





load_test = load_test.reshape(-1,1)

X_test = []
y_test = []
for i in range(7, 960):
    X_test.append(load_test[i-7:i, 0])
    y_test.append(load_test[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test.shape
y_test.shape
 
y_pred=y_pred.reshape(-1,)



y_test

y_pred
y_pred.shape


































"""Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split

num_last_values = 960
total_samples = len(X)

test_indices = range(total_samples - num_last_values, total_samples)

# Split the data into train and test sets using the defined test indices
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=len(test_indices), random_state=0, shuffle=False)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("y_test:", list(y_test))

m = len(time) - 960

load_train = load[:m]
load_test = load[m:]

print("Load_train shape:",load_train.shape)
print("Load_test shape:", load_test.shape)



import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

X_train.shape

from tensorflow.keras.layers import BatchNormalization, Dropout

# Define the LSTM model with optimizations
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM((50), return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM((50), return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(50))
model.add(Dropout(0.4))
model.add(Dense(1))

# Compile the model with a lower learning rate and use early stopping
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# # Implement early stopping
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Train the model with early stopping
# model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
# Train the model with early stopping
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Assuming your model is already trained and you have test data X_test

# Make predictions on the test data
y_pred = model.predict(X_test)

# The predictions will be probabilities since you're using a sigmoid activation in the output layer.
# If you want binary predictions, you can round the probabilities to 0 or 1.
binary_predictions = (y_pred > 0.5).astype('int')

# Optionally, you can evaluate the performance of your model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

y_pred.shape



y_test.shape

y_pred=y_pred.reshape(-1,)

y_test=y_test.reshape(-1,)





comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

df1 = pd.DataFrame(y_test)
df2 = pd.DataFrame(y_pred)

result_df = pd.concat([df1, df2], axis=1)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import mean_squared_error
# Plot original data and predictions
# plt.plot(load, label='Original Load')
plt.plot(y_pred, label="Predicted" )
plt.plot(y_test,color='red',label="Actual")
# plt.plot(np.arange(len(load), len(load)+n_steps), predictions, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('Linear Regression Model Predictions for Load')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define the start and end indices for the portion of data to plot
start_index = 96*4
end_index = 96*6


plt.plot(y_pred[start_index:end_index], label="Predicted")
plt.plot(y_test[start_index:end_index], color='red', label="Actual")

plt.xlabel('Time')
plt.ylabel('Load')
plt.title('Linear Regression Model Predictions for Load for 2 days')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

def mean_absolute_percentage_error(y_test, y_pred):
    return np.mean(np.abs((y_pred - y_test) / y_test)) * 100


mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error (MAPE):", mape)

def mean_absolute_percentage_error(y_test, y_pred):
    return np.mean(np.abs((y_pred - y_test) / y_test)) * 100


mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error (MAPE):", mape)

# Get the slope (coefficients) and intercept of the linear regression model
coefficients = model.coef_
intercept = model.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Form the equation of the line
equation_parts = [f"{coefficients[i]:.2f}x{i+1}" for i in range(len(coefficients))]
equation = " + ".join(equation_parts) + f" + {intercept:.2f}"
print("Equation of the line:", equation)

coefficients = model.coef_
intercept =model.intercept_

# Display coefficients
for i in range(len(coefficients)):
    print(f"Coefficient for x{i+1}: {coefficients[i]}")

print("Intercept:", intercept)



from sklearn.metrics import mean_squared_error
# Plot original data and predictions
# plt.plot(load, label='Original Load')
plt.plot(z_remaining, label="Scheduled" )
plt.plot(y_test,color='red',label="Actual")
# plt.plot(np.arange(len(load), len(load)+n_steps), predictions, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('Scheduled vs Load')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


start_index = 96*4
end_index = 96*6

# Plot selected portion of original data and predictions
plt.plot(z_remaining[start_index:end_index], label="Scheduled")
plt.plot(y_test[start_index:end_index], color='red', label="Actual")

plt.xlabel('Time')
plt.ylabel('Load')
plt.title('Scheduled vs Load for 2 days')
plt.legend()
plt.show()

def mean_absolute_percentage_error(y_test, z_remaining ):
    return np.mean(np.abs((z_remaining - y_test) / y_test)) * 100

# Assuming y_true and y_pred are the true and predicted values, respectively
mape = mean_absolute_percentage_error(y_test, z_remaining )
print("Mean Absolute Percentage Error (MAPE) for load vs scheduled:", mape)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, z_remaining)
r2 = r2_score(y_test,z_remaining)

print("Mean Squared Error:", mse)
print("R-squared:", r2)









import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=18, activation='relu'))
ann.add(tf.keras.layers.Dense(units=18, activation='relu'))
ann.add(tf.keras.layers.Dense(units=18, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))
ann.compile(optimizer = 'RMSProp', loss = 'mean_squared_error',metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size = 32, epochs = 10)

loss, accuracy = ann.evaluate(X, y)
print("Loss:", loss)
print("Accuracy:", accuracy)

y_pred = ann.predict(X_test)
#np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

df3 = pd.DataFrame(y_pred)
df4 = pd.DataFrame(z_remaining)

result_df1 = pd.concat([result_df,df3, df4], axis=1)

df5 = pd.DataFrame(time_remaining)

result_df2 = pd.concat([result_df1, df5], axis=1)

new_column_names = ['Actual Load', 'Prediction_LR', 'Prediction_ANN', 'Scheduled', 'Timestamps']

result_df2.columns = new_column_names

print(result_df2)

result_df2

result_df2.to_csv('combined_data.csv', index=False)



from sklearn.metrics import mean_squared_error
# Plot original data and predictions
# plt.plot(load, label='Original Load')
plt.plot(y_pred, label="Predicted" )
plt.plot(y_test,color='red',label="Actual")
# plt.plot(np.arange(len(load), len(load)+n_steps), predictions, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('ANN Model Predictions for Load')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define the start and end indices for the portion of data to plot
start_index = 96*4
end_index = 96*6

# Plot selected portion of original data and predictions
plt.plot(y_pred[start_index:end_index], label="Predicted")
plt.plot(y_test[start_index:end_index], color='red', label="Actual")

plt.xlabel('Time')
plt.ylabel('Load')
plt.title('ANN Model Predictions for Load for 2 days')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

import numpy as np

def calculate_mape(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ape = np.abs((y_true - y_pred) / y_true)

    ape[np.isnan(ape)] = 0

    mape = np.mean(ape) * 100

    return mape

mape = calculate_mape(y_test, y_pred)
print("MAPE:", mape)

import numpy as np

def calculate_mae(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    absolute_errors = np.abs(y_true - y_pred)

    mae = np.mean(absolute_errors)

    return mae

mae = calculate_mae(y_test, y_pred)
print("MAE:", mae)

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

# Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error (MAPE):", mape)





















value = 13.676

indices = np.where(load == value)

print("Indices of value", value, ":", indices[0])

load[-193]





from statsmodels.tsa.stattools import adfuller

def check_stationarity(series):
    result = adfuller(series)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")



check_stationarity(load)

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
width = 10
height = 6
lag_acf = 170
lag_pacf = 170
f, ax = plt.subplots(nrows=2, ncols=1, figsize=(width, 2*height))

# Plot ACF
plot_acf(load, lags=lag_acf, ax=ax[0])

# Plot PACF
plot_pacf(load, lags=lag_pacf, ax=ax[1], method='ols')

ax[1].annotate('Strong correlation at lag = 1', xy=(1, 0.6), xycoords='data',
               xytext=(0.17, 0.75), textcoords='axes fraction',
               arrowprops=dict(color='red', shrink=0.05, width=1))

plt.tight_layout()
plt.show()

# Create subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

# Plot PACF on the first subplot
plot_pacf(load, ax=ax1)
ax1.set_title('Partial Autocorrelation Function (PACF)')

# Plot ACF on the second subplot
plot_acf(load, ax=ax2)
ax2.set_title('Autocorrelation Function (ACF)')

plt.tight_layout()
plt.show()

load

plt.figure(figsize=(10, 6))
plt.plot(load)
plt.title('Load Data')
plt.xlabel('Quarterly Time')
plt.ylabel('Load in MW')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(load[:700])  # Plotting the first 1000 data points
plt.title('Load Data (First 1000 Data Points)')
plt.xlabel('Quarterly Time')
plt.ylabel('Load in MW')
plt.grid(True)
plt.show()

# Assuming 'load' is your NumPy array containing time series data
dates = pd.date_range(start='2022-01-01', periods=len(load), freq='15min')
load_df = pd.DataFrame({'Value': load}, index=dates)

# Convert index to DatetimeIndex
load_df.index = pd.to_datetime(load_df.index)
load_df

from statsmodels.tsa.seasonal import seasonal_decompose

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Assuming 'load' is your time series data
# Perform seasonal decomposition
decomposition = seasonal_decompose(load, model='additive', period=12)  # Adjust period as needed

# Plot the original time series and its components
plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.plot(load, label='Original')
plt.legend()

plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend()

plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend()

plt.subplot(414)
plt.plot(decomposition.resid, label='Residual')
plt.legend()

plt.tight_layout()
plt.show()



from statsmodels.tsa.seasonal import seasonal_decompose
# Multiplicative Decomposition
plt.figure(figsize=(12, 8))
decomp_mul = seasonal_decompose(df['Load'], model='multiplicative', extrapolate_trend='freq', period=365)
decomp_mul.plot()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
# Multiplicative Decomposition
decomp_mul = seasonal_decompose(df['Load'], model='additive', extrapolate_trend='freq', period=365)
decomp_mul.plot()
plt.show()













"""Split dataset load into Train and Test:
Testing for last 7 days
"""

train=load[:len(load)-100]
test=load[len(load)-100:]

from statsmodels.tsa.ar_model import AutoReg

# Fit AR model
model = AutoReg(train, lags=8).fit() # Example: AR model with lag order 2
print(model.summary())

pred=model.predict(start=len(train),end=len(load)-1,dynamic=False)

comparison = pd.DataFrame({'Actual': test, 'Predicted': pred})
print(comparison)

from sklearn.metrics import mean_squared_error
# Plot original data and predictions
# plt.plot(load, label='Original Load')
plt.plot(pred)
plt.plot(test,color='red')
# plt.plot(np.arange(len(load), len(load)+n_steps), predictions, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('AR Model Predictions for Load')
plt.legend()
plt.show()

# Calculate RMSE (optional)
# test_data = np.random.randn(n_steps)  # Example: test data
rmse = np.sqrt(mean_squared_error(test, pred))
print('Root Mean Squared Error (RMSE):', rmse)

pred_future=model.predict(start=len(load)+1,end=len(load)+7,dynamic=False)
print(pred_future)

from statsmodels.tsa.arima.model import ARIMA
# Fit a Moving Average (MA) model with order 1, discarding initial observations
model = ARIMA(train, order=(0, 0, 5)).fit()  # ARIMA(p=0, d=0, q=1) for MA(1)
print(model.summary())

predMA=model.predict(start=len(train),end=len(load)-1,dynamic=False)

comparison = pd.DataFrame({'Actual': test, 'Predicted': predMA})
print(comparison)
from sklearn.metrics import mean_squared_error
# Plot original data and predictions
# plt.plot(load, label='Original Load')
plt.plot(predMA)
plt.plot(test,color='red')
# plt.plot(np.arange(len(load), len(load)+n_steps), predictions, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('AR Model Predictions for Load')
plt.legend()
plt.show()

# Calculate RMSE (optional)
# test_data = np.random.randn(n_steps)  # Example: test data
rmse = np.sqrt(mean_squared_error(test, predMA))
print('Root Mean Squared Error (RMSE):', rmse)

print(model.summary())

!pip install pmdarima

from pmdarima.arima import auto_arima

model_arima = auto_arima(train, seasonal=False, trace=True)

# Print the summary of the best model
print(model_arima.summary())

from statsmodels.tsa.arima.model import ARIMA
# Fit a Moving Average (MA) model with order 1, discarding initial observations
modelARIMA = ARIMA(train, order=(5, 0, 5)).fit()  # ARIMA(p=0, d=0, q=1) for MA(1)
print(modelARIMA.summary())

predARIMA=modelARIMA.predict(start=len(train),end=len(load)-1,dynamic=False)

comparison = pd.DataFrame({'Actual': test, 'Predicted': predARIMA})
print(comparison)
from sklearn.metrics import mean_squared_error
# Plot original data and predictions
# plt.plot(load, label='Original Load')
plt.plot(test, label='Actual Load')  # Plot the actual load
plt.plot(predARIMA, color='red', label='Predicted Load')  # Plot the predicted load
# plt.plot(np.arange(len(load), len(load)+n_steps), predictions, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('AR Model Predictions for Load')
plt.legend()
plt.show()

# Calculate RMSE (optional)
# test_data = np.random.randn(n_steps)  # Example: test data
rmse = np.sqrt(mean_squared_error(test, predARIMA))
print('Root Mean Squared Error (RMSE):', rmse)

model_arima = auto_arima(train, seasonal=False,start_p=1, d=0, start_q=1,  # Lower values for p and q
                         max_p=5, max_q=5,              # Upper limit for p and q
                         information_criterion='aic',   # Prioritize faster model selection
                          trace=True)

# Print the summary of the best model
print(model_arima.summary())

from statsmodels.tsa.arima.model import ARIMA
# Fit a Moving Average (MA) model with order 1, discarding initial observations
modelARIMA = ARIMA(train, order=(5, 0, 5)).fit()  # ARIMA(p=0, d=0, q=1) for MA(1)
print(modelARIMA.summary())

predARIMA=modelARIMA.predict(start=len(train),end=len(load)-1,dynamic=False)
comparison = pd.DataFrame({'Actual': test, 'Predicted': predARIMA})
print(comparison)
from sklearn.metrics import mean_squared_error
plt.plot(test, label='Actual Load')  # Plot the actual load
plt.plot(predARIMA, color='red', label='Predicted Load')  # Plot the predicted load
# plt.plot(np.arange(len(load), len(load)+n_steps), predictions, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('AR Model Predictions for Load')
plt.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(test, predARIMA))
print('Root Mean Squared Error (RMSE):', rmse)

