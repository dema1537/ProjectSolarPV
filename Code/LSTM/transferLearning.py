import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

import scipy
import matplotlib.pyplot as plt
import time

from torch.utils.data import DataLoader
from sklearn.metrics import precision_score

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import tensorflow as tf
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
from io import StringIO

import os

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

filepath = "Code\LSTM\LSTMModelOutputs\\"

df_weather = pd.read_csv("Data\\facade_weather_data.csv")
df_radiance = pd.read_csv("Data\\facade_solar_data.csv")

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


print("HEads")

df_radiance.reset_index()
df_weather.reset_index()


print(df_weather.head())
print(df_radiance.head())

print("tails")


print(df_weather.tail())
print(df_radiance.tail())

# print(f"Number of NaN values: {df_weather.isna().sum()}")
# print(f"Number of NaN values: {df_radiance.isna().sum()}")

# print(f"Number of all values: {len(df_weather)}")
# print(f"Number of all values: {len(df_radiance)}")

# pd.set_option('display.max_rows', None)
# #print(df_weather)
# # print(df_radiance)
# pd.reset_option('display.max_rows')

# df_weather.to_csv('full_dataframe.csv', index=False)




# df_radiance = df_radiance[1944:]

# df_weather['time'] = pd.to_datetime(df_weather['time'])

# df_radiance.drop(columns=['time'], inplace=True)

# df_weather.insert(0, 'ID', range(1, len(df_weather) + 1))
# df_radiance.insert(0, 'ID', range(1, len(df_radiance) + 1))

df_merged = pd.merge(df_weather, df_radiance, on='ID')

df_merged.drop(columns=['ID'], inplace=True)

df_merged.dropna(inplace=True)

print(df_merged.head())

df_merged.to_csv('full_dataframe.csv', index=False)


#df_np = df_merged[:1000].to_numpy()
df_np = df_merged[:-2].to_numpy()
X = df_np

split_train = int(len(X) * 0.7)
split_val = int(len(X) * 0.85)

X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]

# Sequence Data Preparation
SEQUENCE_SIZE = 5

def to_sequences(seq_size, obs):
    x = []
    y = []
    for i in range(len(obs) - seq_size):
        window = [row[:].copy() for row in obs[i:(i + seq_size)]]
        after_window = obs[i + seq_size-1][34]
        y.append(after_window)

        window[-1][43] = 0.0
        x.append(window)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# def to_sequences(seq_size, obs):
#     x = []
#     y = []
#     for i in range(len(obs) - seq_size):
#         window = [row[:].copy() for row in obs[i:(i + seq_size)]]
#         window_7th = [row[7] for row in window]
#         after_window = obs[i + seq_size][7]
#         y.append(after_window)

#         #window = window[i:i + seq_size][7]
#         x.append(window_7th)
#     return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

xtrain, ytrain = to_sequences(SEQUENCE_SIZE, X_train)
xtest, ytest = to_sequences(SEQUENCE_SIZE, X_test)
xval, yval = to_sequences(SEQUENCE_SIZE, X_val)


# Store original shape before reshaping
xtrain_shape = xtrain.shape
xval_shape = xval.shape
xtest_shape = xtest.shape

# Reshape to 2D for scaling (flatten sequence dimension)
xtrain = xtrain.reshape(-1, xtrain.shape[-1])
xval = xval.reshape(-1, xval.shape[-1])
xtest = xtest.reshape(-1, xtest.shape[-1])

# Fit & transform
scaler_x = MinMaxScaler()
xtrain = scaler_x.fit_transform(xtrain)
xval = scaler_x.transform(xval)
xtest = scaler_x.transform(xtest)

# Reshape back to original 3D shape
xtrain = xtrain.reshape(xtrain_shape)
xval = xval.reshape(xval_shape)
xtest = xtest.reshape(xtest_shape)

ytrain_shape = ytrain.shape
yval_shape = yval.shape
ytest_shape = ytest.shape

ytrain = ytrain.reshape(-1, 1)
yval = yval.reshape(-1, 1)
ytest = ytest.reshape(-1, 1)

scaler_y = MinMaxScaler()
ytrain = scaler_y.fit_transform(ytrain)
ytest = scaler_y.transform(ytest)
yval = scaler_y.transform(yval)


ytrain = ytrain.reshape(ytrain_shape)
yval = yval.reshape(yval_shape)
ytest = ytest.reshape(ytest_shape)

ytrain_shape = ytrain.shape
yval_shape = yval.shape
ytest_shape = ytest.shape



x_train_tensor = torch.tensor(xtrain, dtype=torch.float32)
y_train_tensor = torch.tensor(ytrain, dtype=torch.float32)

x_test_tensor = torch.tensor(xtest, dtype=torch.float32)
y_test_tensor = torch.tensor(ytest, dtype=torch.float32)

x_val_tensor = torch.tensor(xval, dtype=torch.float32)
y_val_tensor = torch.tensor(yval, dtype=torch.float32)

print(x_train_tensor.shape)
print(y_train_tensor.shape)

# Setup data loaders for batch
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)


class LSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(

            nn.LSTM(15, 64, batch_first=True, dropout=0.2),
            nn.Tanh(),

            nn.LSTM(64, 32, batch_first=True, dropout=0.2),
            nn.Tanh(),

            nn.LSTM(32,16, batch_first=True),

            nn.Flatten(),         
        )

        self.classify = nn.Sequential(


            #1
            nn.Linear(16 * 5, 16 * 5),
            nn.ReLU(),
            nn.Linear(16 * 5, 1),
            
        )

    def forward(self, x):
        lstm1_out, _ = self.feature[0](x)
        tanh1_out = self.feature[1](lstm1_out)
        lstm2_out, _ = self.feature[2](tanh1_out)
        tanh2_out = self.feature[3](lstm2_out)
        lstm3_out, _ = self.feature[4](tanh2_out)
        flattened_out = self.feature[5](lstm3_out)
        return self.classify(flattened_out)
        



class newLSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(

            nn.LSTM(54, 64, batch_first=True, dropout=0.2),
            nn.Tanh(),

            nn.LSTM(64, 32, batch_first=True, dropout=0.2),
            nn.Tanh(),

            nn.LSTM(32,16, batch_first=True),

            nn.Flatten(),         
        )

        self.classify = nn.Sequential(


            #1
            nn.Linear(16 * 5, 16 * 5),
            nn.ReLU(),
            nn.Linear(16 * 5, 1),
            
        )

    def forward(self, x):
        lstm1_out, _ = self.feature[0](x)
        tanh1_out = self.feature[1](lstm1_out)
        lstm2_out, _ = self.feature[2](tanh1_out)
        tanh2_out = self.feature[3](lstm2_out)
        lstm3_out, _ = self.feature[4](tanh2_out)
        flattened_out = self.feature[5](lstm3_out)
        return self.classify(flattened_out)


oldModel = LSTM().to(device)

filepath = "Code\LSTM\LSTMModelOutputs\\"

#C:\Users\dw1537\ProjectSolarPV\Code\LSTM\LSTMModelOutputs\best_model.pth
oldModel.load_state_dict(torch.load(filepath + "best_model.pth")) 

newModel = newLSTM().to(device)

old_state_dict = oldModel.state_dict()
new_state_dict = newModel.state_dict()

filtered_dict = {k: v for k, v in old_state_dict.items()
                 if k in new_state_dict and v.shape == new_state_dict[k].shape}

new_state_dict.update(filtered_dict)
newModel.load_state_dict(new_state_dict)


lossFunction = nn.MSELoss()
optimiser = optim.Adam(newModel.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=3, verbose=True)

history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'train_rmse': [],
    'val_rmse': [],
    'train_mape': [],
    'val_mape': []
}

epochs = 50
patience = 10
best_val_loss = 99999999.0
epochs_no_improve = 0
for epoch in range(epochs):
    newModel.train()
    epoch_loss = 0
    total_rmse = 0
    total_mape = 0
    num_batches = 0
    total_samples = 0

    train_losses, train_rmses, train_mapes = [], [], []


    for batch in train_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # optimizer.zero_grad()
        # outputs = classifier(x_batch)
        # loss = criterion(outputs, y_batch)
        # loss.backward()
        # optimizer.step()

        optimiser.zero_grad()  
        predictions = newModel(x_batch).flatten()  

        loss = lossFunction(predictions, y_batch.flatten()) 
        loss.backward() 
        optimiser.step()
        
        batchsize = y_batch.size(0)
        epoch_loss += loss.item() * batchsize  # Scale loss by batch size
        total_samples += batchsize 

        predictions_numpy = predictions.detach().cpu().numpy().reshape(-1, 1)
        y_batch_numpy = y_batch.detach().cpu().numpy().reshape(-1, 1)

        # predictions_original = scaler_y.inverse_transform(predictions_numpy).flatten()
        # y_batch_original = scaler_y.inverse_transform(y_batch_numpy).flatten()

        predictions_original = predictions_numpy.flatten()
        y_batch_original = y_batch_numpy.flatten()

        rmse = np.sqrt(np.mean((predictions_original - y_batch_original) ** 2))
        
        mape = np.mean(np.abs((predictions_original - y_batch_original) / (y_batch_original + 1))) * 100

        total_rmse += rmse
        total_mape += mape
        num_batches += 1

    # Validation
    newModel.eval()
    val_loss = 0
    val_rmse = 0
    val_mape = 0
    val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            val_predictions = newModel(x_batch).flatten()
            loss = lossFunction(val_predictions, y_batch.flatten())
            val_loss += loss.item()

            val_predictions_numpy = val_predictions.cpu().numpy().reshape(-1, 1)
            y_val_numpy = y_batch.cpu().numpy().reshape(-1, 1)

            val_predictions_original = val_predictions_numpy.flatten()
            y_val_original = y_val_numpy.flatten()

            val_rmse += np.sqrt(np.mean((val_predictions_original - y_val_original) ** 2))
            val_mape += np.mean(np.abs((val_predictions_original - y_val_original) / (y_val_original + 1))) * 100 

            val_batches += 1

    avg_rmse = total_rmse / num_batches
    avg_mape = total_mape / num_batches

    avg_val_rmse = val_rmse / val_batches
    avg_val_loss = val_loss / val_batches
    avg_val_mape = val_mape / val_batches

    
    history["train_loss"].append(epoch_loss/total_samples)
    history["train_rmse"].append(avg_rmse)
    history["train_mape"].append(avg_mape)

    history["val_loss"].append(avg_val_loss)
    history["val_rmse"].append(avg_val_rmse)
    history["val_mape"].append(avg_val_mape)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/total_samples:.6f}, Train RMSE: {avg_rmse:.4f}, Train MAPE: {avg_mape:.2f}% | "
          f"Val Loss: {avg_val_loss:.6f}, Val RMSE: {avg_val_rmse:.4f}, Val MAPE: {avg_val_mape:.2f}%")
    
    scheduler.step(avg_val_loss)

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(newModel.state_dict(), filepath + 'TLbest_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            newModel.load_state_dict(torch.load(filepath + 'TLbest_model.pth'))
            break

newModel.load_state_dict(torch.load(filepath + 'TLbest_model.pth'))



newModel.eval()
predictions = []
cloud_cover = []
with torch.no_grad():
    for batch in test_loader:
        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        outputs = newModel(x_batch)
        predictions.extend(outputs.squeeze().tolist())


# for i in range(0, len(xtest)):
#             #0.7634408602150538
#         #78 is my prediction
#         temp = (scaler_x.inverse_transform(np.array(xtest[i][4]).reshape(1, -1)))

#         #print(temp)
#         #print(temp[0][4])
#         # temp = (scaler_x.inverse_transform(np.array(xtest[1][4]).reshape(1, -1)))
#         # print(temp[0][4])
#         cloud_cover.append(temp[0][4])

#100,9,0,5,8,

rmse = np.sqrt(np.mean((scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)) - scaler_y.inverse_transform(ytest.reshape(-1, 1)))**2))
print(f"Score (RMSE): {rmse:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(scaler_y.inverse_transform(ytrain.reshape(-1, 1)).flatten(), label='Actual', color='blue')


test_predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue')
plt.plot(test_predictions, label='Predicted', color='orange')

plt.xlabel("Time Steps")
plt.ylabel("PV")
plt.title("Actual vs Predicted PV")
plt.legend(loc='upper left')
plt.grid(True)
plt.show()


epochs_range = range(1, len(history["val_loss"]) + 1)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(epochs_range, history["val_loss"], label="Val Loss")
plt.plot(epochs_range, history["train_loss"], label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")

plt.subplot(1, 3, 2)
plt.plot(epochs_range, history["val_rmse"], label="Val RMSE")
plt.plot(epochs_range, history["train_rmse"], label="Train RMSE")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.legend()
plt.title("Training vs Validation RMSE")

plt.subplot(1, 3, 3)
plt.plot(epochs_range, history["val_mape"], label="Val MAPE")
plt.plot(epochs_range, history["train_mape"], label="Train MAPE")
plt.xlabel("Epochs")
plt.ylabel("MAPE")
plt.legend()
plt.title("Training vs Validation MAPE")


plt.savefig(filepath + 'TLTrainingValidationGraphs.png')
plt.show()
plt.show()
