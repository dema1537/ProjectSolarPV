from torch.utils.data import DataLoader
from sklearn.metrics import precision_score

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


filename = "CNN.pth.tar"
device = "cpu"

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



split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dayTestX = X[split+110:split+135]
dayTesty = y[split+110:split+135]

df_cloudy = df_weather[(df_weather['cloud_cover (%)'] >= 10) & (df_weather['cloud_cover (%)'] <= 90)]

df_cloudy = pd.merge(df_cloudy, df_radiance, on='ID')

df_cloudy.reset_index(drop=True, inplace=True)

X_train = np.array(X_train)
X_test = np.array(X_test)
dayTestX = np.array(dayTestX)

y_train = np.array(y_train)
y_test = np.array(y_test)
dayTesty = np.array(dayTesty)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

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

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1,1))
y_day_scaled = scaler_y.transform(dayTesty.reshape(-1,1))

batchSize = 64

class TST(nn.Module):

    def __init__(self):
        super(TST, self).__init__()
        self.model = TST(n_in=15, # Number of features in the input
                                           n_out=1,  # Output dimension
                                           d_model=64,  # Dimension of the model's internal representations
                                           n_head=4,  # Number of attention heads
                                           n_layer=4,  # Number of transformer layers
                                           dropout=0.1,
                                           batch_first=True)
        
    def forward(self, x):
        return self.model(x)



classifier = TST().to(device)

lossFunction = nn.MSELoss()

optimiser = optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-3)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)


batch_size = 256
num_samples = X_train_tensor.shape[0]

epochs = 250
for epoch in range(epochs):
    classifier.train()  
    epoch_loss = 0
    total_rmse = 0
    total_mape = 0
    num_batches = 0

    for i in range(0, num_samples, batch_size):
        X_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]

        optimiser.zero_grad()  
        predictions = classifier(X_batch).flatten()  

        loss = lossFunction(predictions, y_batch.flatten()) 
        loss.backward() 
        optimiser.step()
        
        epoch_loss += loss.item()

        # Convert predictions and targets back to original scale
        predictions_numpy = predictions.detach().cpu().numpy().reshape(-1, 1)
        y_batch_numpy = y_batch.detach().cpu().numpy().reshape(-1, 1)

        # predictions_original = scaler_y.inverse_transform(predictions_numpy).flatten()
        # y_batch_original = scaler_y.inverse_transform(y_batch_numpy).flatten()

        predictions_original = predictions_numpy.flatten()
        y_batch_original = y_batch_numpy.flatten()

        # Compute RMSE and MAPE
        rmse = np.sqrt(np.mean((predictions_original - y_batch_original) ** 2))
        
        # Avoid division by very small numbers
        mape = np.mean(np.abs((predictions_original - y_batch_original) / (y_batch_original + 1))) * 100

        total_rmse += rmse
        total_mape += mape
        num_batches += 1

    avg_rmse = total_rmse / num_batches
    avg_mape = total_mape / num_batches

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_samples:.6f}, RMSE: {avg_rmse:.4f}, MAPE: {avg_mape:.2f}%")


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

train_predictions_scaled = classifier(X_train_scaled).flatten()
train_predictions_numpy = train_predictions_scaled.detach().cpu().numpy()
train_predictions = scaler_y.inverse_transform(train_predictions_numpy.reshape(-1, 1)).flatten()

train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})
pd.set_option('display.max_colwidth', 500)
print(train_results)

X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32, requires_grad=False)
test_predictions_scaled = classifier(X_test_scaled).flatten()
test_predictions_numpy = test_predictions_scaled.detach().cpu().numpy()
test_predictions = scaler_y.inverse_transform(test_predictions_numpy.reshape(-1, 1)).flatten()

plt.plot(y_test, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.legend()
plt.show()
