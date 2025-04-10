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
filepath = "Code/LSTM/Outputs/"

if not os.path.exists(filepath):
    os.makedirs(filepath, exist_ok=True)
    print(f"Directory '{filepath}' created successfully.")

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

#data pipeline below

train_loader, val_loader, test_loader, ThreeDayMixed_loader, TwoDaySunny_loader, ThreeDayFull_loader, scaler_y, scaler_x, ytest, xtest, xThreeDayMixed, yThreeDayMixed, xTwoDaySunny, yTwoDaySunny, xThreeDayFull, yThreeDayFull = TrainingDataPipeline()

#training below

history, epochsRan, TrainTime = TrainingLoop(classifier, epochs=3, trainLoader=train_loader, valLoader=val_loader, filepath=filepath, device=device)


classifier.load_state_dict(torch.load(filepath + 'best_model.pth'))
classifier.eval()


#below is testing


Overall_cloud_cover, Overall_predictions, testTime = TestingLoop(classifier, epochsRan, testloader=test_loader, filepath=filepath, device=device, dataSplit=xtest, scaler_x=scaler_x)

#Evaluation and graph metrics  metrics below

#Evaluation of 3 day mixed
ThreeDayMixedCloudCover, ThreeDayMixedPredictions, MixedTime = TestingLoop(classifier, epochsRan, testloader=ThreeDayMixed_loader, filepath=filepath + "DayGraphs/" + "Mixed/", device=device, dataSplit=xThreeDayMixed, scaler_x=scaler_x) 
Evalutaion(ThreeDayMixedPredictions, ThreeDayMixedCloudCover, scaler_y, yThreeDayMixed, history, filepath=filepath + "DayGraphs/" + "Mixed/", title="3 Days with Mixed Cloud Cover")

#Evaluation of 3 day full
ThreeDayFullCloudCover, ThreeDayFullPredictions, FullTime = TestingLoop(classifier, epochsRan, testloader=ThreeDayFull_loader, filepath=filepath + "DayGraphs/" + "Full/", device=device, dataSplit=xThreeDayFull, scaler_x=scaler_x) 
Evalutaion(ThreeDayFullPredictions, ThreeDayFullCloudCover, scaler_y, yThreeDayFull, history, filepath=filepath + "DayGraphs/" + "Full/", title="3 Days with Full cloud cover")

#Evaluation of 2 day sunny
TwoDaySunnyCloudCover, TwoDaySunnyPredictions, SunnyTime = TestingLoop(classifier, epochsRan, testloader=TwoDaySunny_loader, filepath=filepath + "DayGraphs/" + "Sunny/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 
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


with open(filepath + "TimeInfo.txt", "w") as f:
    f.write(f"Train Time: {TrainTime:.3f} secs\n")
    f.write(f"Test Time*: {testTime:.3f} secs\n")
    f.write(f"3 day Mixed Time: {MixedTime:.3f} secs\n")
    f.write(f"3 day Full Time: {FullTime:.3f} secs\n")        
    f.write(f"Sunny Time: {SunnyTime:.3f} secs\n")
print("Time Metrics saved")