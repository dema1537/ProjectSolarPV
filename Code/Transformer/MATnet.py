import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adagrad, Adam

df_weather = pd.read_csv("Data\OpenMeteoData.csv")
df_radiance = pd.read_csv("Data\PVGISdata.csv")

df_radiance = df_radiance[1944:]

df_weather['time'] = pd.to_datetime(df_weather['time'])

df_radiance.drop(columns=['time'], inplace=True)

df_weather.insert(0, 'ID', range(1, len(df_weather) + 1))
df_radiance.insert(0, 'ID', range(1, len(df_radiance) + 1))

df_merged = pd.merge(df_weather, df_radiance, on='ID')

df_merged.drop(columns=['ID'], inplace=True)
df_merged.drop(columns=['time'], inplace=True)

df_merged.dropna(inplace=True)

df_np = df_merged[:].to_numpy()
X = []
y = []

size = 5

# for i in range(len(df_np)-size):
#     input = [a.copy() for a in df_np[i:i + size]]

#     output = []
#     for j in input:
#         output.append(j[7].copy())
#         j[7] = j[7] + 1


#     input = np.delete(input, 7, axis=1)
#     X.append(input)
#     #output = df_np[8][i:i + size]
#     # output = output[8]
#     y.append(output)

for i in range(len(df_np) - size):
    input = [a.copy() for a in df_np[i:i + size]]
    input[size - 1][7] = 0
    X.append(input)
    output = df_np[i + size - 1]
    output = output[7]
    y.append(output)



split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dayTestX = X[split+110:split+135]
dayTesty = y[split+110:split+135]

df_cloudy = df_weather[(df_weather['cloud_cover'] >= 10) & (df_weather['cloud_cover'] <= 90)]

df_cloudy = pd.merge(df_cloudy, df_radiance, on='ID')



# Reset index for convenience (optional)
df_cloudy.reset_index(drop=True, inplace=True)

X_train = np.array(X_train)
X_test = np.array(X_test)
dayTestX = np.array(dayTestX)

y_train = np.array(y_train)
y_test = np.array(y_test)
dayTesty = np.array(dayTesty)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Scale X_train and X_test
scaler_X = MinMaxScaler()
X_train_flat = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
X_test_flat = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
X_day_flat = dayTestX.reshape((dayTestX.shape[0] * dayTestX.shape[1], dayTestX.shape[2]))

X_train_scaled_flat = scaler_X.fit_transform(X_train_flat)
X_test_scaled_flat = scaler_X.transform(X_test_flat)
X_day_flat_flat = scaler_X.transform(X_day_flat)

X_train_scaled = X_train_scaled_flat.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test_scaled = X_test_scaled_flat.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
X_day_flat = X_day_flat_flat.reshape((dayTestX.shape[0], dayTestX.shape[1], dayTestX.shape[2]))

# Scale y_train and y_test
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1,1))
y_day_scaled = scaler_y.transform(dayTesty.reshape(-1,1))





train_predictions_scaled = model1.predict(X_train_scaled).flatten()
train_predictions = scaler_y.inverse_transform(train_predictions_scaled.reshape(-1,1)).flatten()

train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})
pd.set_option('display.max_colwidth', 500)
print(train_results)

test_predictions_scaled = model1.predict(X_test_scaled).flatten()
test_predictions = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1,1)).flatten()

# Plot results
plt.plot(y_test, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.legend()
plt.show()

test_predictions_scaled = model1.predict(X_day_flat).flatten()
test_predictions = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1,1)).flatten()

plt.plot(dayTesty, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.legend()
plt.show()


print("Done")