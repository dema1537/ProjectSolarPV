import torch, math
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.modules.transformer import TransformerEncoderLayer
from torch import nn, Tensor
import torch.nn.functional as F

import os

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")


#ProjectSolarPV\Code\Transformer\ForecastingModel.py
filepath = "Code/Transformer/ForecastingModelOutputs/"


if not os.path.exists(filepath):
    os.makedirs(filepath, exist_ok=True)
    print(f"Directory '{filepath}' created successfully.")

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

#df_np = df_merged[:100].to_numpy()
df_np = df_merged[:-3].to_numpy()
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
        after_window = obs[i + seq_size-1][7]
        y.append(after_window)

        window[-1][7] = 0.0
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

    def forward(self, x: Tensor) -> Tensor:
        #print(x.shape)
        x = x + self.pe[:x.size(1), :, :]
        #print(x.shape)

        return self.dropout(x)

# A forcasting model
class ForecastingModel(torch.nn.Module):
    def __init__(self, 
                 seq_len=5,
                 embed_size = 16,
                 nhead = 4,
                 dim_feedforward = 2048,
                 dropout = 0.1,
                 conv1d_emb = True,
                 conv1d_kernel_size = 5,
                 device = "cpu"):
        super(ForecastingModel, self).__init__()

        # Set Class-level Parameters
        self.device = device
        self.conv1d_emb = conv1d_emb
        self.conv1d_kernel_size = conv1d_kernel_size
        self.seq_len = seq_len
        self.embed_size = embed_size

        # Input Embedding Component
        if conv1d_emb:
            if conv1d_kernel_size%2==0:
                raise Exception("conv1d_kernel_size must be an odd number to preserve dimensions.")
            self.conv1d_padding = conv1d_kernel_size - 1
            self.input_embedding  = nn.Conv1d(15, embed_size, kernel_size=conv1d_kernel_size)
        else: self.input_embedding  = nn.Linear(15, embed_size)

        # Positional Encoder Componet (See Code Copied from PyTorch Above)
        self.position_encoder = PositionalEncoding(d_model=embed_size, 
                                                   dropout=dropout,
                                                   max_len=seq_len)
        
        # Transformer Encoder Layer Component
        self.transformer_encoder = TransformerEncoderLayer(
            d_model = embed_size,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = True
        )

        # Regression Component
        self.linear1 = nn.Linear(seq_len*embed_size, int(dim_feedforward))
        self.linear2 = nn.Linear(int(dim_feedforward), int(dim_feedforward/2))
        self.linear3 = nn.Linear(int(dim_feedforward/2), int(dim_feedforward/4))
        self.linear4 = nn.Linear(int(dim_feedforward/4), int(dim_feedforward/16))
        self.linear5 = nn.Linear(int(dim_feedforward/16), int(dim_feedforward/64))
        self.outlayer = nn.Linear(int(dim_feedforward/64), 1)

        # Basic Components
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # Model Forward Pass
    def forward(self, x):
        src_mask = self._generate_square_subsequent_mask()
        src_mask.to(self.device)
        if self.conv1d_emb:
            #print(x.shape)
 
            x = F.pad(x, (0, 0, self.conv1d_padding, 0), "constant", -1)
            x = self.input_embedding(x.transpose(1, 2)).transpose(1, 2)
            #print(x.shape)
            #x = x.transpose(1, 2)
        else: 
            x = self.input_embedding(x)
        x = self.position_encoder(x)
        #print(x.shape)

        x = self.transformer_encoder(x, src_mask=src_mask).reshape((-1, self.seq_len*self.embed_size))
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear5(x)
        x = self.relu(x)
        return self.outlayer(x)
    
    # Function Copied from PyTorch Library to create upper-triangular source mask
    def _generate_square_subsequent_mask(self):
        return torch.triu(
            torch.full((self.seq_len, self.seq_len), float('-inf'), dtype=torch.float32, device=self.device),
            diagonal=1,
        )




classifier = ForecastingModel().to(device)

# Train the model
lossFunction = nn.MSELoss()
optimiser = torch.optim.Adam(classifier.parameters(), lr=0.0001)
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

epochs = 150
patience = 10
best_val_loss = 99999999.0
epochs_no_improve = 0
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

        optimiser.zero_grad()  
        predictions = classifier(x_batch).flatten()  

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
    classifier.eval()
    val_loss = 0
    val_rmse = 0
    val_mape = 0
    val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
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
    
    scheduler.step(avg_val_loss)

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(classifier.state_dict(), filepath + 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            epochs = epoch 
            classifier.load_state_dict(torch.load(filepath + 'best_model.pth'))
            break

classifier.load_state_dict(torch.load(filepath + 'best_model.pth'))

classifier.eval()
test_loss = 0
test_rmse = 0
test_mape = 0
test_batches = 0
predictions = []
cloud_cover = []
with torch.no_grad():
    for batch in test_loader:
        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        outputs = classifier(x_batch)
        predictions.extend(outputs.squeeze().tolist())

        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        test_predictions = classifier(x_batch).flatten()
        loss = lossFunction(test_predictions, y_batch.flatten())
        test_loss += loss.item()

        test_predictions_numpy = test_predictions.cpu().numpy().reshape(-1, 1)
        y_test_numpy = y_batch.cpu().numpy().reshape(-1, 1)

        test_predictions_original = test_predictions_numpy.flatten()
        y_test_original = y_test_numpy.flatten()

        test_rmse += np.sqrt(np.mean((test_predictions_original - y_test_original) ** 2))
        test_mape += np.mean(np.abs((test_predictions_original - y_test_original) / (y_test_original + 1))) * 100 

        test_batches += 1


avg_test_rmse = test_rmse / test_batches
avg_test_loss = test_loss / test_batches
avg_test_mape = test_mape / test_batches


print("----\n\n\n")
print(f"Test Loss: {avg_test_loss:.6f}, Test RMSE: {avg_test_rmse:.4f}, Test MAPE: {avg_test_mape:.2f}%")



print("\n\n\n----")

with open(filepath + "ForecastingTestMetrics.txt", "w") as f:
    f.write(f"Test Loss: {avg_test_loss:.6f}\n")
    f.write(f"Test RMSE: {avg_test_rmse:.4f}\n")
    f.write(f"Test MAPE: {avg_test_mape:.2f}%\n")
    f.write(f"Test epoch: {epochs:.2f}%\n")
print("Metrics saved")


for i in range(0, len(xtest)):
            #0.7634408602150538
        #78 is my prediction
        temp = (scaler_x.inverse_transform(np.array(xtest[i][4]).reshape(1, -1)))

        #print(temp)
        #print(temp[0][4])
        # temp = (scaler_x.inverse_transform(np.array(xtest[1][4]).reshape(1, -1)))
        # print(temp[0][4])
        cloud_cover.append(temp[0][4])

#100,9,0,5,8,

rmse = np.sqrt(np.mean((scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)) - scaler_y.inverse_transform(ytest.reshape(-1, 1)))**2))
print(f"Score (RMSE): {rmse:.4f}")



xtrain, ytrain2 = to_sequences(SEQUENCE_SIZE, X_train)

X_train_scaled = torch.tensor(xtrain, dtype=torch.float32, requires_grad=False)

# train_predictions_scaled = classifier(x_train).flatten()
# train_predictions_numpy = train_predictions_scaled.detach().cpu().numpy()
# train_predictions = scaler_y.inverse_transform(train_predictions_numpy.reshape(-1, 1)).flatten()



# predictions_train = []
# print(X_train_scaled.shape)
# outputs = classifier(X_train_scaled)
# predictions_train.extend(outputs.squeeze().tolist())
# predictions_train_np = np.array(predictions_train).reshape(-1, 1)
# train_predictions = scaler_y.inverse_transform((predictions_train_np))
# train_results = pd.DataFrame(data={'Train Predictions': train_predictions.flatten(), 'Actuals': scaler_y.inverse_transform(ytrain.reshape(-1, 1)).flatten()})
# pd.set_option('display.max_colwidth', 500)
# print(train_results)

# X_test_scaled = torch.tensor(xtest, dtype=torch.float32, requires_grad=False)
# test_predictions_scaled = classifier(X_test_scaled).flatten()
# test_predictions_numpy = test_predictions_scaled.detach().cpu().numpy()
# test_predictions = scaler_y.inverse_transform(test_predictions_numpy.reshape(-1, 1)).flatten()

test_predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))

# plt.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual')
# plt.plot(test_predictions, label='Predicted')
# plt.legend()
# #plt.show()

# numbers = np.array(cloud_cover)
# plt.bar(range(len(numbers)), numbers)
# plt.ylabel("Cloud Cover (%)")
# #plt.show()

fig, ax2 = plt.subplots()
ax1 = ax2.twinx()

ax1.bar(range(len(cloud_cover)), cloud_cover, label='Cloud Cover', alpha=0.5, width=1.0)
ax1.set_ylabel("Cloud Cover (%)")



ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue')
ax2.plot(test_predictions, label='Predicted', color='orange')
ax2.set_ylabel("PV", color='blue') # Label the second y-axis
ax2.tick_params(axis='y', labelcolor='blue')

ax2.legend(loc='upper left')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

ax1.set_xlabel("Time Steps")

plt.title("Actual and Predicted Values on Mixed Cloud Cover")

ax1.set_xlim(75, 150)
plt.savefig(filepath + '3DayMixedClouds.png')
#plt.show()

fig, ax2 = plt.subplots()
ax1 = ax2.twinx()

ax1.bar(range(len(cloud_cover)), cloud_cover, label='Cloud Cover', alpha=0.5, width=1.0) # Added alpha for better visibility
ax1.set_ylabel("Cloud Cover (%)")



ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue')
ax2.plot(test_predictions, label='Predicted', color='orange')
ax2.set_ylabel("PV", color='blue') # Label the second y-axis
ax2.tick_params(axis='y', labelcolor='blue')

ax2.legend(loc='upper left')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

ax1.set_xlabel("Time Steps")

plt.title("Actual and Predicted Values on Sunny days")


ax1.set_xlim(790, 840)
plt.savefig(filepath + '2DaySunny.png')
#plt.show()

fig, ax2 = plt.subplots()
ax1 = ax2.twinx()

ax1.bar(range(len(cloud_cover)), cloud_cover, label='Cloud Cover', alpha=0.5, width=1.0) # Added alpha for better visibility
ax1.set_ylabel("Cloud Cover (%)")



ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue')
ax2.plot(test_predictions, label='Predicted', color='orange')
ax2.set_ylabel("PV", color='blue') # Label the second y-axis
ax2.tick_params(axis='y', labelcolor='blue')

ax2.legend(loc='upper left')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

ax1.set_xlabel("Time Steps")

plt.title("Actual and Predicted Values on Full Cloud Cover")

ax1.set_xlim(2040, 2110)
ax2.set_ylim(-1000, 22500)
plt.savefig(filepath + '3DayFullCloud.png')
#plt.show()




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


plt.savefig(filepath + 'TrainingValidationGraphs.png')
#plt.show()
#plt.show()
