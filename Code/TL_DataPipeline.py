import pandas as pd

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler



def TL_DataPipeline():

    df_weather = pd.read_csv("Data/facade_weather_data.csv")
    df_radiance = pd.read_csv("Data/facade_solar_data.csv")


    cloud_cover = pd.read_csv("Data/CloudDataForRealWorld.csv")



    df_radiance.drop(columns=['inverterMode'], inplace=True)
    df_radiance.drop(columns=['inverter'], inplace=True)

    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    df_weather.set_index('timestamp', inplace=True)
    df_weather = df_weather.resample('H').mean()
    df_weather.reset_index(inplace=True)
    df_weather = df_weather[412:]

    print(df_weather.head())


    df_radiance['timestamp'] = pd.to_datetime(df_radiance['timestamp'])
    df_radiance.set_index('timestamp', inplace=True)
    df_radiance = df_radiance.resample('H').mean()
    df_radiance.reset_index(inplace=True)
    print(df_radiance[df_radiance['timestamp'] == "2024-10-10 14:00:00"].index.tolist())
    print(df_radiance[df_radiance['timestamp'] == "2025-01-25 12:00:00"].index.tolist())

    df_radiance = df_radiance[8361:10927 + 1]
    df_radiance = df_radiance.fillna(0)

    df_weather.insert(0, 'ID', range(1, len(df_weather) + 1))
    df_radiance.insert(0, 'ID', range(1, len(df_radiance) + 1))

    df_radiance.drop(columns=['timestamp'], inplace=True)
    df_weather.drop(columns=['timestamp'], inplace=True)

    cloud_cover.drop(columns=['time'], inplace=True)

    cloud_cover = cloud_cover.values.flatten()


    print("HEads")

    df_radiance.reset_index()
    df_weather.reset_index()


    print(df_weather.head())
    print(df_radiance.head())

    print("tails")


    print(df_weather.tail())
    print(df_radiance.tail())


    df_merged = pd.merge(df_weather, df_radiance, on='ID')

    df_merged.drop(columns=['ID'], inplace=True)

    df_merged.dropna(inplace=True)

    print(df_merged.head())

    # df_merged.to_csv('full_dataframe.csv', index=False)


    #df_np = df_merged[:1000].to_numpy()
    df_np = df_merged[:-2].to_numpy()
    X = df_np

    split_train = int(len(X) * 0.7)
    split_val = int(len(X) * 0.85)

    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]

    cloud_cover = cloud_cover[:-2]


    print(X_test[0])
    print(cloud_cover[0])


    print(X_test[-1])
    print(cloud_cover[-1])

    # FullCloud = X_test[790:845]
    # TwoDaySunny = X_test[2035:2115]
    # ThreeDayMixed = X_test[70:155]

    FullCloud = X_test[230:300]
    TwoDaySunny = X_test[5:55]

    


    # Sequence Data Preparation
    SEQUENCE_SIZE = 5

    def to_sequences(seq_size, obs):
        x = []
        y = []
        for i in range(len(obs) - seq_size):
            window = [row[:].copy() for row in obs[i:(i + seq_size)]]
            after_window = obs[i + seq_size-1][49]
            y.append(after_window)

            window[-1][49] = 0.0
            x.append(window)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    xtrain, ytrain = to_sequences(SEQUENCE_SIZE, X_train)
    xtest, ytest = to_sequences(SEQUENCE_SIZE, X_test)
    xval, yval = to_sequences(SEQUENCE_SIZE, X_val)

    xFullCloud, yFullCloud = to_sequences(SEQUENCE_SIZE, FullCloud)
    xTwoDaySunny, yTwoDaySunny = to_sequences(SEQUENCE_SIZE, TwoDaySunny)


    # Store original shape before reshaping
    xtrain_shape = xtrain.shape
    xval_shape = xval.shape
    xtest_shape = xtest.shape

    xFullCloud_shape = xFullCloud.shape
    xTwoDaySunny_shape = xTwoDaySunny.shape

    # Reshape to 2D for scaling (flatten sequence dimension)
    xtrain = xtrain.reshape(-1, xtrain.shape[-1])
    xval = xval.reshape(-1, xval.shape[-1])
    xtest = xtest.reshape(-1, xtest.shape[-1])

    xFullCloud = xFullCloud.reshape(-1, xFullCloud.shape[-1])
    xTwoDaySunny = xTwoDaySunny.reshape(-1, xTwoDaySunny.shape[-1])

    # Fit & transform
    scaler_x = MinMaxScaler()
    xtrain = scaler_x.fit_transform(xtrain)
    xval = scaler_x.transform(xval)
    xtest = scaler_x.transform(xtest)

    xFullCloud = scaler_x.transform(xFullCloud)
    xTwoDaySunny = scaler_x.transform(xTwoDaySunny)

    # Reshape back to original 3D shape
    xtrain = xtrain.reshape(xtrain_shape)
    xval = xval.reshape(xval_shape)
    xtest = xtest.reshape(xtest_shape)

    xFullCloud = xFullCloud.reshape(xFullCloud_shape)
    xTwoDaySunny = xTwoDaySunny.reshape(xTwoDaySunny_shape)


    #store shape
    ytrain_shape = ytrain.shape
    yval_shape = yval.shape
    ytest_shape = ytest.shape

    yFullCloud_shape = yFullCloud.shape
    yTwoDaySunny_shape = yTwoDaySunny.shape


    #reshape
    ytrain = ytrain.reshape(-1, 1)
    yval = yval.reshape(-1, 1)
    ytest = ytest.reshape(-1, 1)

    yFullCloud = yFullCloud.reshape(-1, 1)
    yTwoDaySunny = yTwoDaySunny.reshape(-1, 1)


    #scale
    scaler_y = MinMaxScaler()
    ytrain = scaler_y.fit_transform(ytrain)
    ytest = scaler_y.transform(ytest)
    yval = scaler_y.transform(yval)

    yFullCloud = scaler_y.transform(yFullCloud)
    yTwoDaySunny = scaler_y.transform(yTwoDaySunny)

    #reshape to original
    ytrain = ytrain.reshape(ytrain_shape)
    yval = yval.reshape(yval_shape)
    ytest = ytest.reshape(ytest_shape)

    yFullCloud = yFullCloud.reshape(yFullCloud_shape)
    yTwoDaySunny = yTwoDaySunny.reshape(yTwoDaySunny_shape)

    # ytrain_shape = ytrain.shape
    # yval_shape = yval.shape
    # ytest_shape = ytest.shape

    x_train_tensor = torch.tensor(xtrain, dtype=torch.float32)
    y_train_tensor = torch.tensor(ytrain, dtype=torch.float32)

    x_test_tensor = torch.tensor(xtest, dtype=torch.float32)
    y_test_tensor = torch.tensor(ytest, dtype=torch.float32)

    x_val_tensor = torch.tensor(xval, dtype=torch.float32)
    y_val_tensor = torch.tensor(yval, dtype=torch.float32)





    xFullCloud_tensor = torch.tensor(xFullCloud, dtype=torch.float32)
    yFullCloud_tensor = torch.tensor(yFullCloud, dtype=torch.float32)

    xTwoDaySunny_tensor = torch.tensor(xTwoDaySunny, dtype=torch.float32)
    yTwoDaySunny_tensor = torch.tensor(yTwoDaySunny, dtype=torch.float32)


    # Setup data loaders for batch
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)


    FullCloud_dataset = TensorDataset(xFullCloud_tensor, yFullCloud_tensor)
    FullCloud_loader = DataLoader(FullCloud_dataset, batch_size=5, shuffle=False)

    TwoDaySunny_dataset = TensorDataset(xTwoDaySunny_tensor, yTwoDaySunny_tensor)
    TwoDaySunny_loader = DataLoader(TwoDaySunny_dataset, batch_size=5, shuffle=False)

    return train_loader, val_loader, test_loader, FullCloud_loader, TwoDaySunny_loader, scaler_y, scaler_x, ytest, xtest, xFullCloud, yFullCloud, xTwoDaySunny, yTwoDaySunny, cloud_cover, ytrain