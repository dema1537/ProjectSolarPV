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
#import tensorflow as tf
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau



import os

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

filepath = "Code\LSTM\LSTMModelOutputs\\"

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


for i in range(len(df_np) - size):
    input = [a.copy() for a in df_np[i:i + size]]
    input[size - 1][7] = 0
    X.append(input)
    output = df_np[i + size - 1]
    output = output[7]
    y.append(output)



split_train = int(len(X) * 0.7)  
split_val = int(len(X) * 0.85)

X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]
cloudyDayTestX = []
cloudyDayTesty = []

cloudy_indices = df_merged[(df_merged['cloud_cover (%)'] > 20) & (df_merged['cloud_cover (%)'] < 90)].index


for i in range(0, len(X_test)):
    
    if((X_test[i][4][4] > 20) & (X_test[i][4][4] < 90)):
        cloudyDayTestX.append(X_test[i])
        cloudyDayTesty.append(y_test[i])

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
cloudyDayTestX = np.array(cloudyDayTestX)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)
cloudyDayTesty = np.array(cloudyDayTesty)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

scaler_X = MinMaxScaler()
X_train_flat = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
X_val_flat = X_val.reshape((X_val.shape[0] * X_val.shape[1], X_val.shape[2]))
X_test_flat = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
X_day_flat = cloudyDayTestX.reshape((cloudyDayTestX.shape[0] * cloudyDayTestX.shape[1], cloudyDayTestX.shape[2]))

X_train_scaled_flat = scaler_X.fit_transform(X_train_flat)
X_val_scaled_flat = scaler_X.transform(X_val_flat)
X_test_scaled_flat = scaler_X.transform(X_test_flat)
X_day_flat_flat = scaler_X.transform(X_day_flat)

X_train_scaled = X_train_scaled_flat.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_val_scaled = X_val_scaled_flat.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]))
X_test_scaled = X_test_scaled_flat.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
X_day_scaled = X_day_flat_flat.reshape((cloudyDayTestX.shape[0], cloudyDayTestX.shape[1], cloudyDayTestX.shape[2]))

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1))
y_val_scaled = scaler_y.transform(y_val.reshape(-1,1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1,1))
y_day_scaled = scaler_y.transform(cloudyDayTesty.reshape(-1,1))

batchSize = 64

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
        



classifier = LSTM().to(device)

lossFunction = nn.MSELoss()
optimiser = optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=3, verbose=True)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

batch_size = 256
num_samples = X_train_tensor.shape[0]

history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'train_rmse': [],
    'val_rmse': [],
    'train_mape': [],
    'val_mape': []
}

epochs = 25
for epoch in range(epochs):
    classifier.train()  
    epoch_loss = 0
    total_rmse = 0
    total_mape = 0
    num_batches = 0

    train_losses, train_rmses, train_mapes = [], [], []


    for i in range(0, num_samples, batch_size):
        X_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]

        optimiser.zero_grad()  
        predictions = classifier(X_batch).flatten()  

        loss = lossFunction(predictions, y_batch.flatten()) 
        loss.backward() 
        optimiser.step()
        
        epoch_loss += loss.item()

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


    
    classifier.eval()
    val_loss = 0
    val_rmse = 0
    val_mape = 0
    val_batches = 0

    with torch.no_grad():  
        for i in range(0, len(X_val_tensor), batch_size):
            X_val_batch = X_val_tensor[i:i + batch_size]
            y_val_batch = y_val_tensor[i:i + batch_size]

            val_predictions = classifier(X_val_batch).flatten()
            loss = lossFunction(val_predictions, y_val_batch.flatten())
            val_loss += loss.item()

            val_predictions_numpy = val_predictions.cpu().numpy().reshape(-1, 1)
            y_val_numpy = y_val_batch.cpu().numpy().reshape(-1, 1)

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

    history["train_loss"].append(epoch_loss/num_samples)
    history["train_rmse"].append(avg_rmse)
    history["train_mape"].append(avg_mape)

    history["val_loss"].append(avg_val_loss)
    history["val_rmse"].append(avg_val_rmse)
    history["val_mape"].append(avg_val_mape)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/num_samples:.6f}, Train RMSE: {avg_rmse:.4f}, Train MAPE: {avg_mape:.2f}% | "
          f"Val Loss: {avg_val_loss:.6f}, Val RMSE: {avg_val_rmse:.4f}, Val MAPE: {avg_val_mape:.2f}%")


# epochs = 25
# for epoch in range(epochs):
#     classifier.train()  
#     epoch_loss = 0
#     total_rmse = 0
#     total_mape = 0
#     num_batches = 0

#     for i in range(0, num_samples, batch_size):
#         X_batch = X_train_tensor[i:i + batch_size]
#         y_batch = y_train_tensor[i:i + batch_size]

#         optimiser.zero_grad()  
#         predictions = classifier(X_batch) 

#         loss = lossFunction(predictions, y_batch) 
#         loss.backward() 
#         optimiser.step()
#         epoch_loss += loss.item()

#         predictions_numpy = predictions.detach().cpu().numpy()
#         y_batch_numpy = y_batch.detach().cpu().numpy()
#         predictions_original = scaler_y.inverse_transform(predictions_numpy.reshape(-1, 1)).flatten()
#         y_batch_original = scaler_y.inverse_transform(y_batch_numpy.reshape(-1, 1)).flatten()

#         rmse = np.sqrt(np.mean((predictions_original - y_batch_original) ** 2))
#         mape = np.mean(np.abs((predictions_original - y_batch_original) / (y_batch_original + 1e-8))) * 100  # Avoid division by zero

#         total_rmse += rmse
#         total_mape += mape
#         num_batches += 1

#     avg_rmse = total_rmse / num_batches
#     avg_mape = total_mape / num_batches

#     print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_samples:.6f}, RMSE: {avg_rmse:.4f}, MAPE: {avg_mape:.2f}%")


X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32, requires_grad=False)

train_predictions_scaled = torch.clamp(classifier(X_train_scaled).flatten(), min=0)
train_predictions_numpy = train_predictions_scaled.detach().cpu().numpy()
train_predictions = scaler_y.inverse_transform(train_predictions_numpy.reshape(-1, 1)).flatten()

train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})
pd.set_option('display.max_colwidth', 500)
print(train_results)

X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32, requires_grad=False)
test_predictions_scaled = torch.clamp(classifier(X_test_scaled).flatten(), min=0)
test_predictions_numpy = test_predictions_scaled.detach().cpu().numpy()
test_predictions = scaler_y.inverse_transform(test_predictions_numpy.reshape(-1, 1)).flatten()

plt.plot(y_test, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.legend()
plt.show()


#The epoch graphs#

epochs_range = range(1, epochs + 1)

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

plt.show()