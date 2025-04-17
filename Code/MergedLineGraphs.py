import sys
import os
import torch
import numpy as np
import matplotlib as plt


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

import scipy
import matplotlib.pyplot as plt
import time

from torch import nn, Tensor

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

from torch.nn.modules.transformer import TransformerEncoderLayer
import torch.nn.functional as F

dir = "Code/"  

if dir not in sys.path:
    sys.path.append(dir)

dir = "Code/CNN"  

if dir not in sys.path:
    sys.path.append(dir)
    
dir = "Code/LSTM"  

if dir not in sys.path:
    sys.path.append(dir)

from TrainingLoop import TrainingLoop
from TestingLoop import TestingLoop 
from Evaluation import Evalutaion
from TrainingDataPipeline import TrainingDataPipeline
from TL_DataPipeline import TL_DataPipeline


###models####

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



class CNNLSTM(nn.Module):
    def __init__(self, input_size, output_size,hidden_size,num_layers):
        super(CNNLSTM, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(64,32,kernel_size=1, stride = 1, padding=1)
        self.batch1 =nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32,32,kernel_size=1, stride = 1, padding=1)
        self.batch2 =nn.BatchNorm1d(32)
        self.LSTM = nn.LSTM(input_size=18, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(32*hidden_size, output_size)
        #self.fc2 = nn.Linear(1, 1)
        

    def forward(self, x):
        #in_size1 = x.size(0)  # one batch
        x = F.selu(self.conv1(x))
        x = self.conv2(x)
        x = F.selu(self.batch1(x))
        x = self.conv3(x)
        x = F.selu(self.batch2(x))
        x, h = self.LSTM(x) 
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
        #in_size1 = x.size(0)  # one batch
        #x = x.view(in_size1, -1)
        # flatten the tensor x[:, -1, :]
        x = self.fc1(x)
        output = torch.sigmoid(x)
        #output = self.fc2(x)

    
        
        return output






device = "cpu"
filepath = "AdditionalFiles/"

train_loader, val_loader, test_loader, ThreeDayMixed_loader, TwoDaySunny_loader, ThreeDayFull_loader, scaler_y, scaler_x, ytest, xtest, xThreeDayMixed, yThreeDayMixed, xTwoDaySunny, yTwoDaySunny, xThreeDayFull, yThreeDayFull = TrainingDataPipeline()

epochsRan = 999



print("1")

classifier = CNN().to(device)

classifier.load_state_dict(torch.load("Code/CNN/Outputs/" + 'best_model.pth'))
classifier.eval()

print("2")

#Evaluation of 3 day mixed
ThreeDayMixedCloudCover, CNNThreeDayMixedPredictions, MixedTime = TestingLoop(classifier, epochsRan, testloader=ThreeDayMixed_loader, filepath=filepath + "DayGraphs/" + "Mixed/", device=device, dataSplit=xThreeDayMixed, scaler_x=scaler_x) 

print("3")

#Evaluation of 3 day full
ThreeDayFullCloudCover, CNNThreeDayFullPredictions, FullTime = TestingLoop(classifier, epochsRan, testloader=ThreeDayFull_loader, filepath=filepath + "DayGraphs/" + "Full/", device=device, dataSplit=xThreeDayFull, scaler_x=scaler_x) 
print("4")
#Evaluation of 2 day sunny
TwoDaySunnyCloudCover, CNNTwoDaySunnyPredictions, SunnyTime = TestingLoop(classifier, epochsRan, testloader=TwoDaySunny_loader, filepath=filepath + "DayGraphs/" + "Sunny/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 


print("5")




classifier = LSTM().to(device)

classifier.load_state_dict(torch.load("Code/LSTM/Outputs/" + 'best_model.pth'))
classifier.eval()



#Evaluation of 3 day mixed
ThreeDayMixedCloudCover, LSTMThreeDayMixedPredictions, MixedTime = TestingLoop(classifier, epochsRan, testloader=ThreeDayMixed_loader, filepath=filepath + "DayGraphs/" + "Mixed/", device=device, dataSplit=xThreeDayMixed, scaler_x=scaler_x) 

#Evaluation of 3 day full
ThreeDayFullCloudCover, LSTMThreeDayFullPredictions, FullTime = TestingLoop(classifier, epochsRan, testloader=ThreeDayFull_loader, filepath=filepath + "DayGraphs/" + "Full/", device=device, dataSplit=xThreeDayFull, scaler_x=scaler_x) 

#Evaluation of 2 day sunny
TwoDaySunnyCloudCover, LSTMTwoDaySunnyPredictions, SunnyTime = TestingLoop(classifier, epochsRan, testloader=TwoDaySunny_loader, filepath=filepath + "DayGraphs/" + "Sunny/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 





classifier = CNNLSTM(input_size=5, output_size=1,hidden_size=64, num_layers=3).to(device)

classifier.load_state_dict(torch.load("Code/CNNLSTM/Outputs/" + 'best_model.pth'))
classifier.eval()


#Evaluation of 3 day mixed
ThreeDayMixedCloudCover, CNNLSTMThreeDayMixedPredictions, MixedTime = TestingLoop(classifier, epochsRan, testloader=ThreeDayMixed_loader, filepath=filepath + "DayGraphs/" + "Mixed/", device=device, dataSplit=xThreeDayMixed, scaler_x=scaler_x) 

#Evaluation of 3 day full
ThreeDayFullCloudCover, CNNLSTMThreeDayFullPredictions, FullTime = TestingLoop(classifier, epochsRan, testloader=ThreeDayFull_loader, filepath=filepath + "DayGraphs/" + "Full/", device=device, dataSplit=xThreeDayFull, scaler_x=scaler_x) 

#Evaluation of 2 day sunny
TwoDaySunnyCloudCover, CNNLSTMTwoDaySunnyPredictions, SunnyTime = TestingLoop(classifier, epochsRan, testloader=TwoDaySunny_loader, filepath=filepath + "DayGraphs/" + "Sunny/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 





CNNtest_predictions = scaler_y.inverse_transform(np.array(CNNThreeDayMixedPredictions).reshape(-1, 1))
LSTMtest_predictions = scaler_y.inverse_transform(np.array(LSTMThreeDayMixedPredictions).reshape(-1, 1))
CNNLSTMtest_predictions = scaler_y.inverse_transform(np.array(CNNLSTMThreeDayMixedPredictions).reshape(-1, 1))


ytest = yThreeDayMixed





fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()




ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue', lw=4)
ax2.plot(CNNLSTMtest_predictions, label='CNNLSTM Predicted', color='orange', ls="--")
ax2.plot(LSTMtest_predictions, label='LSTM Predicted', color='cyan', ls="--")
ax2.plot(CNNtest_predictions, label='CNN Predicted', color='lightgreen', ls="--")
ax2.set_ylabel("PV", color='blue') 
ax2.tick_params(axis='y', labelcolor='blue')

ax2.legend(loc='upper left')

ax1.bar(range(len(ThreeDayMixedCloudCover)), ThreeDayMixedCloudCover, label='Cloud Cover', alpha=0.5, width=1.0)
ax1.set_ylabel("Cloud Cover (%)")

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

ax1.set_xlabel("Time Steps")

plt.title("Actual and Predicted Values on Mixed Cloud for Baseline Models")

# ax1.set_xlim(75, 150)
plt.savefig(filepath + "Merged3dayMixedCloudGraph" + '.png')

#plt.show()




ytest = yThreeDayFull


CNNtest_predictions = scaler_y.inverse_transform(np.array(CNNThreeDayFullPredictions).reshape(-1, 1))
LSTMtest_predictions = scaler_y.inverse_transform(np.array(LSTMThreeDayFullPredictions).reshape(-1, 1))
CNNLSTMtest_predictions = scaler_y.inverse_transform(np.array(CNNLSTMThreeDayFullPredictions).reshape(-1, 1))




fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()




ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue', lw=4)
ax2.plot(CNNtest_predictions, label='CNN Predicted', color='lightgreen', ls="--")
ax2.plot(CNNLSTMtest_predictions, label='CNNLSTM Predicted', color='orange', ls="--")
ax2.plot(LSTMtest_predictions, label='LSTM Predicted', color='cyan', ls="--")
ax2.set_ylabel("PV", color='blue') 
ax2.tick_params(axis='y', labelcolor='blue')

ax2.legend(loc='upper left')

ax1.bar(range(len(ThreeDayFullCloudCover)), ThreeDayFullCloudCover, label='Cloud Cover', alpha=0.5, width=1.0)
ax1.set_ylabel("Cloud Cover (%)")

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

ax1.set_xlabel("Time Steps")

plt.title("Actual and Predicted Values on Full Cloud for Baseline Models")

# ax1.set_xlim(75, 150)
plt.savefig(filepath + "Merged3dayFullCloudGraph" + '.png')


ytest = yTwoDaySunny


CNNtest_predictions = scaler_y.inverse_transform(np.array(CNNTwoDaySunnyPredictions).reshape(-1, 1))
LSTMtest_predictions = scaler_y.inverse_transform(np.array(LSTMTwoDaySunnyPredictions).reshape(-1, 1))
CNNLSTMtest_predictions = scaler_y.inverse_transform(np.array(CNNLSTMTwoDaySunnyPredictions).reshape(-1, 1))




fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()




ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue', lw=4)
ax2.plot(CNNLSTMtest_predictions, label='CNNLSTM Predicted', color='orange', ls="--")
ax2.plot(LSTMtest_predictions, label='LSTM Predicted', color='cyan', ls="--")
ax2.plot(CNNtest_predictions, label='CNN Predicted', color='lightgreen', ls="--")
ax2.set_ylabel("PV", color='blue') 
ax2.tick_params(axis='y', labelcolor='blue')

ax2.legend(loc='upper left')

ax1.bar(range(len(TwoDaySunnyCloudCover)), TwoDaySunnyCloudCover, label='Cloud Cover', alpha=0.5, width=1.0)
ax1.set_ylabel("Cloud Cover (%)")

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

ax1.set_xlabel("Time Steps")

plt.title("Actual and Predicted Values on Sunny Days for Baseline Models")

# ax1.set_xlim(75, 150)
plt.savefig(filepath + "Merged2DaySunGraph" + '.png')

#plt.show()


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



class newCNNLSTM(nn.Module):
    def __init__(self, input_size, output_size,hidden_size,num_layers):
        super(newCNNLSTM, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(64,32,kernel_size=1, stride = 1, padding=1)
        self.batch1 =nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32,32,kernel_size=1, stride = 1, padding=1)
        self.batch2 =nn.BatchNorm1d(32)
        self.LSTM = nn.LSTM(input_size=57, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(32*hidden_size, output_size)
        #self.fc2 = nn.Linear(1, 1)
        

    def forward(self, x):
        #in_size1 = x.size(0)  # one batch
        x = F.selu(self.conv1(x))
        x = self.conv2(x)
        x = F.selu(self.batch1(x))
        x = self.conv3(x)
        x = F.selu(self.batch2(x))
        x, h = self.LSTM(x) 
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
        #in_size1 = x.size(0)  # one batch
        #x = x.view(in_size1, -1)
        # flatten the tensor x[:, -1, :]
        x = self.fc1(x)
        output = torch.sigmoid(x)
        #output = self.fc2(x)

    
        
        return output


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
    

train_loader, val_loader, test_loader, FullCloud_loader, TwoDaySunny_loader, scaler_y, scaler_x, ytest, xtest, xFullCloud, yFullCloud, xTwoDaySunny, yTwoDaySunny, cloud_cover, ytrain = TL_DataPipeline()

FineTuneEpochs = 20
epochs = 21

classifier = newCNN().to(device)



classifier.load_state_dict(torch.load("Code/CNN/Outputs/" + "Pretrained/"+ 'best_model.pth'))
classifier.eval()

_, CNNTwoDaySunnyPredictions, SunnyTime = TestingLoop(classifier, epochsRan=epochs, testloader=TwoDaySunny_loader, filepath=filepath + "Pretrained/RealDayGraphs/" + "Sunny/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 
# Evalutaion(TwoDaySunnyPredictions, cloud_cover[5:55], scaler_y, yTwoDaySunny, history, filepath=filepath + "Pretrained/RealDayGraphs/" + "Sunny/", title="2 Sunny Days")

_, CNNFullPredictions, CloudyTime = TestingLoop(classifier, epochsRan=epochs, testloader=FullCloud_loader, filepath=filepath + "Pretrained/RealDayGraphs/" + "Cloud/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 
# Evalutaion(FullPredictions, cloud_cover[230:300], scaler_y, yFullCloud, history, filepath=filepath + "Pretrained/RealDayGraphs/" + "Cloud/", title="3 Cloudy Days")

classifier = newLSTM().to(device)



classifier.load_state_dict(torch.load("Code/LSTM/Outputs/" + "Pretrained/"+ 'best_model.pth'))
classifier.eval()

_, LSTMTwoDaySunnyPredictions, SunnyTime = TestingLoop(classifier, epochsRan=epochs, testloader=TwoDaySunny_loader, filepath=filepath + "Pretrained/RealDayGraphs/" + "Sunny/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 
# Evalutaion(TwoDaySunnyPredictions, cloud_cover[5:55], scaler_y, yTwoDaySunny, history, filepath=filepath + "Pretrained/RealDayGraphs/" + "Sunny/", title="2 Sunny Days")

_, LSTMFullPredictions, CloudyTime = TestingLoop(classifier, epochsRan=epochs, testloader=FullCloud_loader, filepath=filepath + "Pretrained/RealDayGraphs/" + "Cloud/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 
# Evalutaion(FullPredictions, cloud_cover[230:300], scaler_y, yFullCloud, history, filepath=filepath + "Pretrained/RealDayGraphs/" + "Cloud/", title="3 Cloudy Days")


classifier = newCNNLSTM(input_size=5, output_size=1,hidden_size=64, num_layers=3).to(device)


classifier.load_state_dict(torch.load("Code/CNNLSTM/Outputs/" + "Pretrained/"+ 'best_model.pth'))
classifier.eval()

_, CNNLSTMTwoDaySunnyPredictions, SunnyTime = TestingLoop(classifier, epochsRan=epochs, testloader=TwoDaySunny_loader, filepath=filepath + "Pretrained/RealDayGraphs/" + "Sunny/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 
# Evalutaion(TwoDaySunnyPredictions, cloud_cover[5:55], scaler_y, yTwoDaySunny, history, filepath=filepath + "Pretrained/RealDayGraphs/" + "Sunny/", title="2 Sunny Days")

_, CNNLSTMFullPredictions, CloudyTime = TestingLoop(classifier, epochsRan=epochs, testloader=FullCloud_loader, filepath=filepath + "Pretrained/RealDayGraphs/" + "Cloud/", device=device, dataSplit=xTwoDaySunny, scaler_x=scaler_x) 
# Evalutaion(FullPredictions, cloud_cover[230:300], scaler_y, yFullCloud, history, filepath=filepath + "Pretrained/RealDayGraphs/" + "Cloud/", title="3 Cloudy Days")





ytest = yTwoDaySunny


CNNtest_predictions = scaler_y.inverse_transform(np.array(CNNTwoDaySunnyPredictions).reshape(-1, 1))
LSTMtest_predictions = scaler_y.inverse_transform(np.array(LSTMTwoDaySunnyPredictions).reshape(-1, 1))
CNNLSTMtest_predictions = scaler_y.inverse_transform(np.array(CNNLSTMTwoDaySunnyPredictions).reshape(-1, 1))




fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()




ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue', lw=4)
ax2.plot(CNNLSTMtest_predictions, label='CNNLSTM Predicted', color='orange', ls="--")
ax2.plot(LSTMtest_predictions, label='LSTM Predicted', color='cyan', ls="--")
ax2.plot(CNNtest_predictions, label='CNN Predicted', color='lightgreen', ls="--")
ax2.set_ylabel("PV", color='blue') 
ax2.tick_params(axis='y', labelcolor='blue')

ax2.legend(loc='upper left')

ax1.bar(range(len(cloud_cover[5:55])), cloud_cover[5:55], label='Cloud Cover', alpha=0.5, width=1.0)
ax1.set_ylabel("Cloud Cover (%)")

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

ax1.set_xlabel("Time Steps")

plt.title("Actual and Predicted Values on Sunny Days for Baseline Models (Real World Data)")

# ax1.set_xlim(75, 150)
plt.savefig(filepath + "RealMerged2DaySunGraph" + '.png')




ytest = yFullCloud


CNNtest_predictions = scaler_y.inverse_transform(np.array(CNNFullPredictions).reshape(-1, 1))
LSTMtest_predictions = scaler_y.inverse_transform(np.array(LSTMFullPredictions).reshape(-1, 1))
CNNLSTMtest_predictions = scaler_y.inverse_transform(np.array(CNNLSTMFullPredictions).reshape(-1, 1))




fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()




ax2.plot(scaler_y.inverse_transform(ytest.reshape(-1, 1)).flatten(), label='Actual', color='blue', lw=4)
ax2.plot(CNNtest_predictions, label='CNN Predicted', color='lightgreen', ls="--")
ax2.plot(CNNLSTMtest_predictions, label='CNNLSTM Predicted', color='orange', ls="--")
ax2.plot(LSTMtest_predictions, label='LSTM Predicted', color='cyan', ls="--")
ax2.set_ylabel("PV", color='blue') 
ax2.tick_params(axis='y', labelcolor='blue')

ax2.legend(loc='upper left')

ax1.bar(range(len(cloud_cover[230:300])), cloud_cover[230:300], label='Cloud Cover', alpha=0.5, width=1.0)
ax1.set_ylabel("Cloud Cover (%)")

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

ax1.set_xlabel("Time Steps")

plt.title("Actual and Predicted Values on Sunny Days for Baseline Models (Real World Data)")

# ax1.set_xlim(75, 150)
plt.savefig(filepath + "RealMergedCloudGraph" + '.png')
