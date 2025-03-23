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

df_cloudy = []
for j in range(0, len(df_merged), 24):

    temp = []
    total = 0
    for k in range(0, 24):
        temp.append(df_merged.iloc[j+k])
        cloud_cover = df_merged.iloc[j+k]['cloud_cover (%)']
        if not pd.isna(cloud_cover):
            total = total + int(cloud_cover)


    if total <= 2000:  
        df_cloudy.append(temp)


df_cloudy = (df_cloudy[551:])



df_merged.drop(columns=['time'], inplace=True)

# for df in df_cloudy:
#     df.drop(columns=['time'], inplace=True)

df_merged.dropna(inplace=True)
df_np = df_merged[:].to_numpy()
X = []
y = []

Xcloudy = []
ycloudy = []

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

temp_df_cloudy = []

# for j in range(0,25):

#     for i in range(len(df_cloudy[j]) - 1 - size):

#         temp = pd.DataFrame(df_cloudy[j]).drop(columns=['time'])
        
#         temp_df_cloudy.append(temp)
#         # temp_df_cloudy[j].drop(columns=['time'], inplace=True)

#         input = [a.copy() for a in temp_df_cloudy[j][i:i + size]]
#         input[i][size - 1][7] = 0
#         Xcloudy.append(input)
#         output = df_cloudy[j].iloc[i + size - 1, 7]
#         ycloudy.append(output)


X1 = np.array(X)
y1 = np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dayTestX = Xcloudy
dayTesty = ycloudy


df_cloudy = pd.merge(df_cloudy, df_radiance, on='ID')

df_cloudy.drop(columns=['ID'], inplace=True)
df_cloudy.drop(columns=['time'], inplace=True)

df_cloudy.dropna(inplace=True)

print(df_cloudy.head())
print(df_cloudy.shape())

cloud_X_test = X[split:]
cloud_y_test = y[split:]

cloud_X_test = np.array(cloud_X_test)
cloud_y_test = np.array(cloud_y_test)

scaler_X = MinMaxScaler()

X_cloud_flat = cloud_X_test.reshape((cloud_X_test.shape[0] * cloud_X_test.shape[1], cloud_X_test.shape[2]))
y_cloud_flat = cloud_y_test.reshape((cloud_y_test.shape[0] * cloud_y_test.shape[1], cloud_y_test.shape[2]))

X_cloud_scaled_flat = scaler_X.transform(X_cloud_flat)
y_cloud_scaled_flat = scaler_X.transform(y_cloud_flat)




