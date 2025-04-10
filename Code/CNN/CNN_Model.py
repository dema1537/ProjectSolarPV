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
filepath = "Code/CNN/test/CNNOutputs/"

if not os.path.exists(filepath):
    os.makedirs(filepath, exist_ok=True)
    print(f"Directory '{filepath}' created successfully.")

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



classifier = CNN().to(device)

#data pipeline below

train_loader, val_loader, test_loader, ThreeDayMixed_loader, scaler_y, scaler_x, ytest, xtest, xThreeDayMixed, yThreeDayMixed = TrainingDataPipeline()

#training below

history, epochsRan = TrainingLoop(classifier, epochs=3, trainLoader=train_loader, valLoader=val_loader, filepath=filepath, device=device)

classifier.load_state_dict(torch.load(filepath + 'best_model.pth'))
classifier.eval()


#below is testing

Overall_cloud_cover, Overall_predictions = TestingLoop(classifier, epochsRan, testloader=test_loader, filepath=filepath, device=device, dataSplit=xtest, scaler_x=scaler_x)


ThreeDayMixedCloudCover, ThreeDayMixedPredictions = TestingLoop(classifier, epochsRan, testloader=ThreeDayMixed_loader, filepath=filepath + "threedaytest", device=device, dataSplit=xThreeDayMixed, scaler_x=scaler_x) 

#Testing above

#Evaluation and graph metrics  metrics below
#Evalutaion(Overall_cloud_cover, Overall_predictions, scaler_y, ytest, history, filepath)

Evalutaion(ThreeDayMixedPredictions, ThreeDayMixedCloudCover, scaler_y, yThreeDayMixed, history, filepath, title="graph title")

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