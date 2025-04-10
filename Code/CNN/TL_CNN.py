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

filepath = "Code/CNN/Outputs/"


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(

            nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3),
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
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),


            #2
            nn.Linear(32, 1),
            nn.ReLU()

        )

    def forward(self, x):
        features = self.feature(x)

        return self.classify(features)
        



class newCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(

            nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3),
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

        Nchannels = self.feature(torch.empty(1, 5, 54)).size(-1)

        self.classify = nn.Sequential(

            #1
            nn.Linear(int(Nchannels), 32),
            nn.ReLU(),
            nn.Dropout(0.5),

            #2
            nn.Linear(32, 1),
            nn.ReLU()

        )

    def forward(self, x):
        features = self.feature(x)

        return self.classify(features)



oldModel = CNN().to(device)

train_loader, val_loader, test_loader, FullCloud_loader, TwoDaySunny_loader, scaler_y, scaler_x, ytest, xtest, xFullCloud, yFullCloud, xTwoDaySunny, yTwoDaySunny, cloud_cover, ytrain = TL_DataPipeline()

oldModel.load_state_dict(torch.load(filepath + "best_model.pth")) 

newModel = newCNN().to(device)

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
oldrunModel = newCNN().to(device)

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

