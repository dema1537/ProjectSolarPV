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

from torch.nn.modules.transformer import TransformerEncoderLayer


import os
from io import StringIO

import torch.nn.functional as F


import os
import sys

dir = "Code/"  

if dir not in sys.path:
    sys.path.append(dir)

from TrainingLoop import TrainingLoop
from TestingLoop import TestingLoop 
from Evaluation import Evalutaion
from TL_DataPipeline import TL_DataPipeline

device = "cpu"
FineTuneEpochs = 20


if torch.cuda.is_available():
    device = "cuda"
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

filepath = "Code/Transformer/ForecastingOutputs/"


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


class newForecastingModel(torch.nn.Module):
    def __init__(self, 
                 seq_len=5,
                 embed_size = 16,
                 nhead = 4,
                 dim_feedforward = 2048,
                 dropout = 0.1,
                 conv1d_emb = True,
                 conv1d_kernel_size = 5,
                 device = "cpu"):
        super(newForecastingModel, self).__init__()

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
            self.input_embedding  = nn.Conv1d(54, embed_size, kernel_size=conv1d_kernel_size)
        else: self.input_embedding  = nn.Linear(54, embed_size)

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


oldModel = ForecastingModel().to(device)

#CNNLSTM(input_size=5, output_size=1,hidden_size=64, num_layers=3).to(device)
train_loader, val_loader, test_loader, FullCloud_loader, TwoDaySunny_loader, scaler_y, scaler_x, ytest, xtest, xFullCloud, yFullCloud, xTwoDaySunny, yTwoDaySunny, cloud_cover, ytrain = TL_DataPipeline()

oldModel.load_state_dict(torch.load(filepath + "best_model.pth")) 

newModel = newForecastingModel().to(device)

old_state_dict = oldModel.state_dict()
new_state_dict = newModel.state_dict()

filtered_dict = {k: v for k, v in old_state_dict.items()
                 if k in new_state_dict and v.shape == new_state_dict[k].shape}

new_state_dict.update(filtered_dict)
newModel.load_state_dict(new_state_dict)


history, epochs, PretrainedFineTuneTime = TrainingLoop(newModel, epochs=FineTuneEpochs, trainLoader=train_loader, valLoader=val_loader, filepath=filepath + "Pretrained/", device=device)

newModel.load_state_dict(torch.load(filepath + "Pretrained/"+ 'best_model.pth'))


_, _, PretrainedTestTime = TestingLoop(newModel, epochsRan=epochs, testloader=TwoDaySunny_loader, filepath=filepath + "Pretrained/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 


_, TwoDaySunnyPredictions, SunnyTime = TestingLoop(newModel, epochsRan=epochs, testloader=TwoDaySunny_loader, filepath=filepath + "Pretrained/RealDayGraphs/" + "Sunny/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 
Evalutaion(TwoDaySunnyPredictions, cloud_cover[5:55], scaler_y, yTwoDaySunny, history, filepath=filepath + "Pretrained/RealDayGraphs/" + "Sunny/", title="2 Sunny Days")

_, FullPredictions, CloudyTime = TestingLoop(newModel, epochsRan=epochs, testloader=FullCloud_loader, filepath=filepath + "Pretrained/RealDayGraphs/" + "Cloud/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 
Evalutaion(FullPredictions, cloud_cover[230:300], scaler_y, yFullCloud, history, filepath=filepath + "Pretrained/RealDayGraphs/" + "Cloud/", title="3 Cloudy Days")

#Epoch 50/50, Train Loss: 0.000400, Train RMSE: 0.0145, Train MAPE: 0.73% | Val Loss: 0.000209, Val RMSE: 0.0088, Val MAPE: 0.63%
oldrunModel = newForecastingModel().to(device)

history, epochs, NonPretrainedFineTuneTime = TrainingLoop(oldrunModel, epochs=FineTuneEpochs, trainLoader=train_loader, valLoader=val_loader, filepath=filepath + "NonPretrained/", device=device)


oldrunModel.load_state_dict(torch.load(filepath + "NonPreTrained/" + 'best_model.pth'))
oldrunModel.eval()


_, _, NontrainedTestTime = TestingLoop(oldrunModel, epochsRan=epochs, testloader=TwoDaySunny_loader, filepath=filepath + "NonPretrained/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 


_, TwoDaySunnyPredictions, SunnyTime = TestingLoop(oldrunModel, epochsRan=epochs, testloader=TwoDaySunny_loader, filepath=filepath + "NonPretrained/RealDayGraphs/" + "Sunny/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 
Evalutaion(TwoDaySunnyPredictions, cloud_cover[5:55], scaler_y, yTwoDaySunny, history, filepath=filepath + "NonPretrained/RealDayGraphs/" + "Sunny/", title="2 Sunny Days")

_, FullPredictions, CloudyTime = TestingLoop(oldrunModel, epochsRan=epochs, testloader=FullCloud_loader, filepath=filepath + "NonPretrained/RealDayGraphs/" + "Cloud/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 
Evalutaion(FullPredictions, cloud_cover[230:300], scaler_y, yFullCloud, history, filepath=filepath + "NonPretrained/RealDayGraphs/" + "Cloud/", title="3 Cloudy Days")


with open(filepath + "TL_TimeInfo.txt", "w") as f:
    f.write(f"PreTrain finetune Time: {PretrainedFineTuneTime:.3f} secs\n")
    f.write(f"Pretrain Test Time*: {PretrainedTestTime:.3f} secs\n")
    f.write(f"NonPreTrain finetune Time: {NonPretrainedFineTuneTime:.3f} secs\n")
    f.write(f"NonPreTrain Test Time: {NontrainedTestTime:.3f} secs\n")        
    f.write(f"Sunny Time: {SunnyTime:.3f} secs\n")
print("TL Time Metrics saved")

