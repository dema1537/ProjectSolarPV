import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import scipy
import matplotlib.pyplot as plt
import time

from torch.utils.data import DataLoader
from sklearn.metrics import precision_score

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adagrad, Adam


filename = "CNN.pth.tar"
device = "cpu"

# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
#     device = "cuda"

# else:
#     print("No GPU available. Training will run on CPU.")



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


batchSize = 64


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(

            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),


            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),
            nn.ReLU(),


            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),


            nn.Flatten(),

            
        )


        self.classify = nn.Sequential(


            #1
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.5),


            #2
            nn.Linear(32, 1),

        )

    def forward(self, x):
        features = self.feature(x)

        return self.classify(features)



classifier = CNN().to(device)

lossFunction = nn.MeanSquaredError()

optimiser = Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-3)

# Assuming you have your X_train, y_train as numpy arrays
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

# Define your batch size and total samples
batch_size = 64
num_samples = X_train_tensor.shape[0]

# Training loop
epochs = 300
for epoch in range(epochs):
    nn.train()  # Set model to training mode
    epoch_loss = 0

    # Iterate over the dataset in mini-batches
    for i in range(0, num_samples, batch_size):
        # Create mini-batch
        X_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]

        # Forward pass
        optimiser.zero_grad()  # Zero the gradients
        predictions = nn(X_batch)  # Forward pass through the model
        loss = lossFunction(predictions, y_batch)  # Calculate loss

        # Backward pass and optimization
        loss.backward()  # Compute gradients
        optimiser.step()  # Update parameters

        epoch_loss += loss.item()

    # Print loss at the end of each epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_samples:.4f}")