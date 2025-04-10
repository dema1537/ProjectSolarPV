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
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, Tensor
from torch.nn.modules.transformer import TransformerEncoderLayer
import torch.nn.functional as F



import sys
import os

dir = "Code/"  

if dir not in sys.path:
    sys.path.append(dir)

from TrainingLoop import TrainingLoop
from TestingLoop import TestingLoop 
from Evaluation import Evalutaion
from TrainingDataPipeline import TrainingDataPipeline

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")


#ProjectSolarPV\Code\Transformer\ForecastingModel.py
filepath = "Code/Transformer/SimpleTransformerOutputs/"

if not os.path.exists(filepath):
    os.makedirs(filepath, exist_ok=True)
    print(f"Directory '{filepath}' created successfully.")

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

#data pipeline below

train_loader, val_loader, test_loader, ThreeDayMixed_loader, TwoDaySunny_loader, ThreeDayFull_loader, scaler_y, scaler_x, ytest, xtest, xThreeDayMixed, yThreeDayMixed, xTwoDaySunny, yTwoDaySunny, xThreeDayFull, yThreeDayFull = TrainingDataPipeline()

#training below

history, epochsRan = TrainingLoop(classifier, epochs=3, trainLoader=train_loader, valLoader=val_loader, filepath=filepath, device=device)

classifier.load_state_dict(torch.load(filepath + 'best_model.pth'))
classifier.eval()


#below is testing

Overall_cloud_cover, Overall_predictions = TestingLoop(classifier, epochsRan, testloader=test_loader, filepath=filepath, device=device, dataSplit=xtest, scaler_x=scaler_x)



#Testing above

#Evaluation and graph metrics  metrics below



#Evaluation of 3 day mixed
ThreeDayMixedCloudCover, ThreeDayMixedPredictions = TestingLoop(classifier, epochsRan, testloader=ThreeDayMixed_loader, filepath=filepath + "DayGraphs/" + "Mixed/", device=device, dataSplit=xThreeDayMixed, scaler_x=scaler_x) 
Evalutaion(ThreeDayMixedPredictions, ThreeDayMixedCloudCover, scaler_y, yThreeDayMixed, history, filepath=filepath + "DayGraphs/" + "Mixed/", title="3 Days with Mixed Cloud COver")

#Evaluation of 3 day full
ThreeDayFullCloudCover, ThreeDayFullPredictions = TestingLoop(classifier, epochsRan, testloader=ThreeDayFull_loader, filepath=filepath + "DayGraphs/" + "Full/", device=device, dataSplit=xThreeDayFull, scaler_x=scaler_x) 
Evalutaion(ThreeDayFullPredictions, ThreeDayFullCloudCover, scaler_y, yThreeDayFull, history, filepath=filepath + "DayGraphs/" + "Full/", title="3 Days with Full cloud cover")

#Evaluation of 2 day sunny
TwoDaySunnyCloudCover, TwoDaySunnyPredictions = TestingLoop(classifier, epochsRan, testloader=TwoDaySunny_loader, filepath=filepath + "DayGraphs/" + "Sunny/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 
Evalutaion(TwoDaySunnyPredictions, TwoDaySunnyCloudCover, scaler_y, yTwoDaySunny, history, filepath=filepath + "DayGraphs/" + "Sunny/", title="2 Sunny Days")


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