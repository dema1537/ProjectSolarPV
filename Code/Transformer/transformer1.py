import torch
from torch import nn
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
import tensorflow as tf
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau



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
X = df_np



split_train = int(len(X) * 0.7)  
split_val = int(len(X) * 0.85)

X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)
X_val = scaler_x.transform(X_val)





# Sequence Data Preparation
SEQUENCE_SIZE = 5

def to_sequences(seq_size, obs):
    x = []
    y = []
    for i in range(len(obs) - seq_size):
        window = [row[:].copy() for row in obs[i:(i + seq_size)]] 
        after_window = obs[i + seq_size][7]
        y.append(after_window)


        window[-1][7] = 0.0
        x.append(window)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

x_train, y_train = to_sequences(SEQUENCE_SIZE, X_train)
x_test, y_test = to_sequences(SEQUENCE_SIZE, X_test)
x_val, y_val = to_sequences(SEQUENCE_SIZE, X_val)

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)
y_val = scaler_y.transform(y_val)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

print(x_train_tensor.shape)
print(y_train_tensor.shape)
# Setup data loaders for batch
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)






# size = 5


# # for i in range(len(df_np) - size):
# #     input = [a.copy() for a in df_np[i:i + size]]
# #     input[size - 1][7] = 0
# #     X.append(input)
# #     output = df_np[i + size - 1]
# #     output = output[7]
# #     y.append(output)



# split_train = int(len(X) * 0.7)  
# split_val = int(len(X) * 0.85)

# X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
# y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]
# cloudyDayTestX = []
# cloudyDayTesty = []

# cloudy_indices = df_merged[(df_merged['cloud_cover (%)'] > 20) & (df_merged['cloud_cover (%)'] < 90)].index


# for i in range(0, len(X_test)):
    
#     if((X_test[i][4][4] > 20) & (X_test[i][4][4] < 90)):
#         cloudyDayTestX.append(X_test[i])
#         cloudyDayTesty.append(y_test[i])

# X_train = np.array(X_train)
# X_val = np.array(X_val)
# X_test = np.array(X_test)
# cloudyDayTestX = np.array(cloudyDayTestX)

# y_train = np.array(y_train)
# y_val = np.array(y_val)
# y_test = np.array(y_test)
# cloudyDayTesty = np.array(cloudyDayTesty)

# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)

# scaler_X = MinMaxScaler()
# X_train_flat = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
# X_val_flat = X_val.reshape((X_val.shape[0] * X_val.shape[1], X_val.shape[2]))
# X_test_flat = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
# X_day_flat = cloudyDayTestX.reshape((cloudyDayTestX.shape[0] * cloudyDayTestX.shape[1], cloudyDayTestX.shape[2]))

# X_train_scaled_flat = scaler_X.fit_transform(X_train_flat)
# X_val_scaled_flat = scaler_X.fit_transform(X_val_flat)
# X_test_scaled_flat = scaler_X.transform(X_test_flat)
# X_day_flat_flat = scaler_X.transform(X_day_flat)

# X_train_scaled = X_train_scaled_flat.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
# X_val_scaled = X_val_scaled_flat.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]))
# X_test_scaled = X_test_scaled_flat.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
# X_day_scaled = X_day_flat_flat.reshape((cloudyDayTestX.shape[0], cloudyDayTestX.shape[1], cloudyDayTestX.shape[2]))

# scaler_y = MinMaxScaler()
# y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1))
# y_val_scaled = scaler_y.fit_transform(y_val.reshape(-1,1))
# y_test_scaled = scaler_y.transform(y_test.reshape(-1,1))
# y_day_scaled = scaler_y.transform(cloudyDayTesty.reshape(-1,1))




# batchSize = 64

# class Transformer(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.feature = nn.Sequential(

#             nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3),
#             nn.Dropout(0.5),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),

#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.ReLU(),


#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
#             nn.Dropout(0.5),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),


#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.ReLU(),


#             nn.Flatten(),

            
#         )


#         self.classify = nn.Sequential(


#             #1
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Dropout(0.5),


#             #2
#             nn.Linear(32, 1),
#             nn.ReLU()

#         )

#     def forward(self, x):
#         features = self.feature(x)

#         return self.classify(features)



# classifier = Transformer().to(device)

# lossFunction = nn.MSELoss()

# optimiser = optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-3)

# X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
# X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
# y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)


# batch_size = 256
# num_samples = X_train_tensor.shape[0]


# history = {
#     'epoch': [],
#     'train_loss': [],
#     'val_loss': [],
#     'train_rmse': [],
#     'val_rmse': [],
#     'train_mape': [],
#     'val_mape': []
# }

# epochs = 25
# for epoch in range(epochs):
#     classifier.train()  
#     epoch_loss = 0
#     total_rmse = 0
#     total_mape = 0
#     num_batches = 0

#     train_losses, train_rmses, train_mapes = [], [], []

#     for i in range(0, num_samples, batch_size):
#         X_batch = X_train_tensor[i:i + batch_size]
#         y_batch = y_train_tensor[i:i + batch_size]

#         optimiser.zero_grad()  
#         predictions = classifier(X_batch).flatten()  

#         loss = lossFunction(predictions, y_batch.flatten()) 
#         loss.backward() 
#         optimiser.step()
        
#         epoch_loss += loss.item()

#         # Convert predictions and targets back to original scale
#         predictions_numpy = predictions.detach().cpu().numpy().reshape(-1, 1)
#         y_batch_numpy = y_batch.detach().cpu().numpy().reshape(-1, 1)

#         # predictions_original = scaler_y.inverse_transform(predictions_numpy).flatten()
#         # y_batch_original = scaler_y.inverse_transform(y_batch_numpy).flatten()

#         predictions_original = predictions_numpy.flatten()
#         y_batch_original = y_batch_numpy.flatten()

#         # Compute RMSE and MAPE
#         rmse = np.sqrt(np.mean((predictions_original - y_batch_original) ** 2))
        
#         # Avoid division by very small numbers
#         mape = np.mean(np.abs((predictions_original - y_batch_original) / (y_batch_original + 1))) * 100

#         total_rmse += rmse
#         total_mape += mape
#         num_batches += 1
    



#     classifier.eval()
#     val_loss = 0
#     val_rmse = 0
#     val_mape = 0
#     val_batches = 0

#     with torch.no_grad():  
#         for i in range(0, len(X_val_tensor), batch_size):
#             X_val_batch = X_val_tensor[i:i + batch_size]
#             y_val_batch = y_val_tensor[i:i + batch_size]

#             val_predictions = classifier(X_val_batch).flatten()
#             loss = lossFunction(val_predictions, y_val_batch.flatten())
#             val_loss += loss.item()

#             val_predictions_numpy = val_predictions.cpu().numpy().reshape(-1, 1)
#             y_val_numpy = y_val_batch.cpu().numpy().reshape(-1, 1)

#             val_predictions_original = val_predictions_numpy.flatten()
#             y_val_original = y_val_numpy.flatten()

#             val_rmse += np.sqrt(np.mean((val_predictions_original - y_val_original) ** 2))
#             val_mape += np.mean(np.abs((val_predictions_original - y_val_original) / (y_val_original + 1))) * 100 

#             val_batches += 1

#     avg_rmse = total_rmse / num_batches
#     avg_mape = total_mape / num_batches

#     avg_val_rmse = val_rmse / val_batches
#     avg_val_loss = val_loss / val_batches
#     avg_val_mape = val_mape / val_batches

#     history["train_loss"].append(epoch_loss/num_samples)
#     history["train_rmse"].append(avg_rmse)
#     history["train_mape"].append(avg_mape)

#     history["val_loss"].append(avg_val_loss)
#     history["val_rmse"].append(avg_val_rmse)
#     history["val_mape"].append(avg_val_mape)



#     print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/num_samples:.6f}, Train RMSE: {avg_rmse:.4f}, Train MAPE: {avg_mape:.2f}% | "
#           f"Val Loss: {avg_val_loss:.6f}, Val RMSE: {avg_val_rmse:.4f}, Val MAPE: {avg_val_mape:.2f}%")

# # epochs = 25
# # for epoch in range(epochs):
# #     classifier.train()  
# #     epoch_loss = 0
# #     total_rmse = 0
# #     total_mape = 0
# #     num_batches = 0

# #     for i in range(0, num_samples, batch_size):
# #         X_batch = X_train_tensor[i:i + batch_size]
# #         y_batch = y_train_tensor[i:i + batch_size]

# #         optimiser.zero_grad()  
# #         predictions = classifier(X_batch) 

# #         loss = lossFunction(predictions, y_batch) 
# #         loss.backward() 
# #         optimiser.step()
# #         epoch_loss += loss.item()

# #         predictions_numpy = predictions.detach().cpu().numpy()
# #         y_batch_numpy = y_batch.detach().cpu().numpy()
# #         predictions_original = scaler_y.inverse_transform(predictions_numpy.reshape(-1, 1)).flatten()
# #         y_batch_original = scaler_y.inverse_transform(y_batch_numpy.reshape(-1, 1)).flatten()

# #         rmse = np.sqrt(np.mean((predictions_original - y_batch_original) ** 2))
# #         mape = np.mean(np.abs((predictions_original - y_batch_original) / (y_batch_original + 1e-8))) * 100  # Avoid division by zero

# #         total_rmse += rmse
# #         total_mape += mape
# #         num_batches += 1

# #     avg_rmse = total_rmse / num_batches
# #     avg_mape = total_mape / num_batches

# #     print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_samples:.6f}, RMSE: {avg_rmse:.4f}, MAPE: {avg_mape:.2f}%")



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=17028):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class TransformerModel(nn.Module):
    def __init__(self, input_dim=15, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x

classifier = TransformerModel().to(device)

# Train the model
lossFunction = nn.MSELoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)


early_stop_count = 0
min_val_loss = float('inf')

history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'train_rmse': [],
    'val_rmse': [],
    'train_mape': [],
    'val_mape': []
}

epochs = 1
for epoch in range(epochs):


    classifier.train()
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

        optimizer.zero_grad()  
        predictions = classifier(x_batch).flatten()  

        loss = lossFunction(predictions, y_batch.flatten()) 
        loss.backward() 
        optimizer.step()
        
        batch_size = y_batch.size(0)
        epoch_loss += loss.item() * batch_size  # Scale loss by batch size
        total_samples += batch_size 

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
    classifier.eval()
    val_loss = 0
    val_rmse = 0
    val_mape = 0
    val_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            val_predictions = classifier(x_batch).flatten()
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

classifier.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        outputs = classifier(x_batch)
        predictions.extend(outputs.squeeze().tolist())

rmse = np.sqrt(np.mean((scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)) - scaler_y.inverse_transform(y_test.reshape(-1, 1)))**2))
print(f"Score (RMSE): {rmse:.4f}")




X_train_scaled = torch.tensor(x_train, dtype=torch.float32, requires_grad=False)

# train_predictions_scaled = classifier(x_train).flatten()
# train_predictions_numpy = train_predictions_scaled.detach().cpu().numpy()
# train_predictions = scaler_y.inverse_transform(train_predictions_numpy.reshape(-1, 1)).flatten()
predictions_train = []
outputs = classifier(X_train_scaled)
predictions_train.extend(outputs.squeeze().tolist())
predictions_train_np = np.array(predictions_train).reshape(-1, 1)
train_predictions = scaler_y.inverse_transform((predictions_train_np))
train_predictions = scaler_y.inverse_transform((train_predictions))
train_predictions = scaler_y.inverse_transform((train_predictions))
train_predictions = scaler_y.inverse_transform((train_predictions))
train_results = pd.DataFrame(data={'Train Predictions': train_predictions.flatten(), 'Actuals': scaler_y.inverse_transform(y_train.numpy().reshape(-1, 1)).flatten()})
pd.set_option('display.max_colwidth', 500)
print(train_results)

X_test_scaled = torch.tensor(x_test, dtype=torch.float32, requires_grad=False)
test_predictions_scaled = classifier(X_test_scaled).flatten()
test_predictions_numpy = test_predictions_scaled.detach().cpu().numpy()
test_predictions = scaler_y.inverse_transform(test_predictions_numpy.reshape(-1, 1)).flatten()

plt.plot(scaler_y.inverse_transform(y_test).flatten(), label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.legend()
plt.show()

epochs_range = range(1, epochs + 1)

# plt.subplot(1, 3, 1)
# plt.plot(epochs_range, history["val_loss"], label="Val Loss")
# plt.plot(epochs_range, history["train_loss"], label="Train Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.title("Training vs Validation Loss")

# plt.subplot(1, 3, 2)
# plt.plot(epochs_range, history["val_rmse"], label="Val RMSE")
# plt.plot(epochs_range, history["train_rmse"], label="Train RMSE")
# plt.xlabel("Epochs")
# plt.ylabel("RMSE")
# plt.legend()
# plt.title("Training vs Validation RMSE")

# plt.subplot(1, 3, 3)
# plt.plot(epochs_range, history["val_mape"], label="Val MAPE")
# plt.plot(epochs_range, history["train_mape"], label="Train MAPE")
# plt.xlabel("Epochs")
# plt.ylabel("MAPE")
# plt.legend()
# plt.title("Training vs Validation MAPE")

# plt.show()
