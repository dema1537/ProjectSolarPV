import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler



def TrainingDataPipeline():

    num = 1000


    num = -3
    df_weather = pd.read_csv("Data/OpenMeteoData.csv")
    df_radiance = pd.read_csv("Data/PVGISdata.csv")

    df_radiance = df_radiance[1944:]

    df_weather['time'] = pd.to_datetime(df_weather['time'])

    df_radiance.drop(columns=['time'], inplace=True)

    df_weather.insert(0, 'ID', range(1, len(df_weather) + 1))
    df_radiance.insert(0, 'ID', range(1, len(df_radiance) + 1))

    df_merged = pd.merge(df_weather, df_radiance, on='ID')

    df_merged.drop(columns=['ID'], inplace=True)
    df_merged.drop(columns=['time'], inplace=True)

    df_merged.dropna(inplace=True)

    df_np = df_merged[:num].to_numpy()
    X = df_np

    split_train = int(len(X) * 0.7)
    split_val = int(len(X) * 0.85)

    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]


    # TwoDaySunny = X_test[785:845]
    # ThreeDayFull = X_test[2035:2115]

    ThreeDayMixed = X_test[70:155]
    


    # Sequence Data Preparation
    SEQUENCE_SIZE = 5

    def to_sequences(seq_size, obs):
        x = []
        y = []
        for i in range(len(obs) - seq_size):
            window = [row[:].copy() for row in obs[i:(i + seq_size)]]
            after_window = obs[i + seq_size-1][7]
            y.append(after_window)

            window[-1][7] = 0.0
            x.append(window)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    xtrain, ytrain = to_sequences(SEQUENCE_SIZE, X_train)
    xtest, ytest = to_sequences(SEQUENCE_SIZE, X_test)
    xval, yval = to_sequences(SEQUENCE_SIZE, X_val)

    xThreeDayMixed, yThreeDayMixed = to_sequences(SEQUENCE_SIZE, ThreeDayMixed)


    # Store original shape before reshaping
    xtrain_shape = xtrain.shape
    xval_shape = xval.shape
    xtest_shape = xtest.shape

    xThreeDayMixed_shape = xThreeDayMixed.shape

    # Reshape to 2D for scaling (flatten sequence dimension)
    xtrain = xtrain.reshape(-1, xtrain.shape[-1])
    xval = xval.reshape(-1, xval.shape[-1])
    xtest = xtest.reshape(-1, xtest.shape[-1])

    xThreeDayMixed = xThreeDayMixed.reshape(-1, xThreeDayMixed.shape[-1])

    # Fit & transform
    scaler_x = MinMaxScaler()
    xtrain = scaler_x.fit_transform(xtrain)
    xval = scaler_x.transform(xval)
    xtest = scaler_x.transform(xtest)

    xThreeDayMixed = scaler_x.transform(xThreeDayMixed)

    # Reshape back to original 3D shape
    xtrain = xtrain.reshape(xtrain_shape)
    xval = xval.reshape(xval_shape)
    xtest = xtest.reshape(xtest_shape)

    xThreeDayMixed = xThreeDayMixed.reshape(xThreeDayMixed_shape)


    #store shape
    ytrain_shape = ytrain.shape
    yval_shape = yval.shape
    ytest_shape = ytest.shape

    yThreeDayMixed_shape = yThreeDayMixed.shape


    #reshape
    ytrain = ytrain.reshape(-1, 1)
    yval = yval.reshape(-1, 1)
    ytest = ytest.reshape(-1, 1)

    yThreeDayMixed = yThreeDayMixed.reshape(-1, 1)


    #scale
    scaler_y = MinMaxScaler()
    ytrain = scaler_y.fit_transform(ytrain)
    ytest = scaler_y.transform(ytest)
    yval = scaler_y.transform(yval)

    yThreeDayMixed = scaler_y.transform(yThreeDayMixed)

    #reshape to original
    ytrain = ytrain.reshape(ytrain_shape)
    yval = yval.reshape(yval_shape)
    ytest = ytest.reshape(ytest_shape)

    yThreeDayMixed = yThreeDayMixed.reshape(yThreeDayMixed_shape)

    # ytrain_shape = ytrain.shape
    # yval_shape = yval.shape
    # ytest_shape = ytest.shape

    x_train_tensor = torch.tensor(xtrain, dtype=torch.float32)
    y_train_tensor = torch.tensor(ytrain, dtype=torch.float32)

    x_test_tensor = torch.tensor(xtest, dtype=torch.float32)
    y_test_tensor = torch.tensor(ytest, dtype=torch.float32)

    x_val_tensor = torch.tensor(xval, dtype=torch.float32)
    y_val_tensor = torch.tensor(yval, dtype=torch.float32)




    xThreeDayMixed_tensor = torch.tensor(xThreeDayMixed, dtype=torch.float32)
    yThreeDayMixed_tensor = torch.tensor(yThreeDayMixed, dtype=torch.float32)


    # Setup data loaders for batch
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    ThreeDayMixed_dataset = TensorDataset(xThreeDayMixed_tensor, yThreeDayMixed_tensor)
    ThreeDayMixed_loader = DataLoader(ThreeDayMixed_dataset, batch_size=5, shuffle=False)

    return train_loader, val_loader, test_loader, ThreeDayMixed_loader, scaler_y, scaler_x, ytest, xtest, xThreeDayMixed, yThreeDayMixed