import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adagrad


df_weather = pd.read_csv("Data\OpenMeteoData.csv")





df_radiance = pd.read_csv("Data\PVGISdata.csv")

df_radiance = df_radiance[1944:]

# print(df_weather.head())
# print(df_radiance.head())

# print(df_radiance.count())
# print(df_weather.count())

df_weather['time'] = pd.to_datetime(df_weather['time'])
# print(df_weather.head())

df_radiance.drop(columns=['time'], inplace=True)

df_weather.insert(0, 'ID', range(1, len(df_weather) + 1))
df_radiance.insert(0, 'ID', range(1, len(df_radiance) + 1))



# print(df_weather.head())
# print(df_radiance.head())

df_merged = pd.merge(df_weather, df_radiance, on='ID')

df_merged.drop(columns=['ID'], inplace=True)
df_merged.drop(columns=['time'], inplace=True)

print(df_merged.isna().sum())
df_merged.dropna(inplace=True)


# print(df_merged.head())

# temp = df_merged['P']
# temp[:23].plot()
# plt.show()


df_np = df_merged[:].to_numpy()
X = []
y = []

size = 14

for i in range(len(df_np)-size):
    input = [a.copy() for a in df_np[i:i + size]]

    output = []
    for j in input:
        output.append(j[7].copy())
        j[7] = j[7] + 1


    input = np.delete(input, 7, axis=1)
    X.append(input)
    #output = df_np[8][i:i + size]
    # output = output[8]
    y.append(output)

X1 = np.array(X)
y1 = np.array(y)



print("X shape is:")
print(X1.shape)

print("y shape is:")
print(y1.shape)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

dayTestX = X[split+100:split+125]
dayTesty = y[split+100:split+125]

X_train = np.array(X_train)
X_test = np.array(X_test)

dayTestX = np.array(dayTestX)

y_train = np.array(y_train)
y_test = np.array(y_test)

dayTesty = np.array(dayTesty)

# scaler = MinMaxScaler(feature_range=(0, 1))
# X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
# X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))


model1 = Sequential()
#model1.add(InputLayer((5, 15)))
model1.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(14,14)))
model1.add(LSTM(32, activation='tanh', return_sequences=False))
model1.add(Dense(16, 'relu'))
model1.add(Dense(14, 'linear'))




# model.compile(optimizer='adam', loss='mse')
model1.compile(loss='mean_absolute_percentage_error', optimizer=Adagrad(learning_rate=0.001), metrics=[RootMeanSquaredError()])

model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=1)

#y_pred =  model.predict(X_test)

train_predictions = model1.predict(X_train)
train_results = pd.DataFrame(data={'Train Predictions':train_predictions.flatten(), 'Actuals':y_train.flatten()})
print(train_results)

# Plot results
plt.plot(y_test, label='Actual')
plt.plot(train_predictions, label='Predicted')
plt.legend()
plt.show()

print("Done")



# print(df_weather.tail())
# print(df_radiance.tail())


